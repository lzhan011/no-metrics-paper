#!/usr/bin/env python3
"""Shared graph-coloring metric helpers.

These helpers mirror the SAT evaluation analysis pipeline but parse
Graph_Colouring_Evaluation prediction outputs and validate coloring
assignments instead of Boolean SAT assignments.
"""

import csv
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ANALYSIS_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, os.pardir))
REPO_ROOT = os.path.abspath(os.path.join(ANALYSIS_DIR, os.pardir))
GROUP_EVAL_ROOT = os.path.abspath(os.path.join(REPO_ROOT, os.pardir))
GRAPH_ROOT = os.path.join(GROUP_EVAL_ROOT, "Graph_Colouring_Evaluation")
SAT_ROOT = REPO_ROOT
SAT_COMMERCIAL_ANALYSIS_DIR = os.path.join(
    SAT_ROOT, "analysis", "commercial_api_results_analysis"
)
if SAT_COMMERCIAL_ANALYSIS_DIR not in sys.path:
    sys.path.insert(0, SAT_COMMERCIAL_ANALYSIS_DIR)

import compute_commercial_api_metrics as commercial_metrics  # noqa: E402
import commercial_api_group_adr_with_wilson_and_cp_RMC as rmc_core  # noqa: E402


SAT = commercial_metrics.SAT
UNSAT = commercial_metrics.UNSAT
DEFAULT_OPEN_SOURCE_ROOT = os.path.join(GRAPH_ROOT, "open_source_LLM_3_Colouring")
DEFAULT_OUTPUT_DIR = SCRIPT_DIR
DEFAULT_OUTPUT_PREFIX = "open_source_LLM_3_Colouring_group_adr_with_wilson_and_cp_RMC_report"
MIN_EDIT_DATASET_DIR = "min_edit_graph_coloring_dataset"

commercial_metrics.MIN_EDIT_DATASET_DIR = MIN_EDIT_DATASET_DIR

STRICT_CLUSTER_HEADERS = [
    "num_clusters_total",
    "num_clusters_all_pairs_complete",
    "num_clusters_all_pairs_success",
    "cluster_all_pairs_complete_rate",
    "cluster_all_pairs_success_rate",
    "cluster_success_gap_ADR_minus_strict_cluster_success",
]

_STRICT_CLUSTER_HEADERS_TO_APPEND = [
    key for key in STRICT_CLUSTER_HEADERS if key not in rmc_core.BY_N_HEADERS
]
BY_N_HEADERS = rmc_core.BY_N_HEADERS + _STRICT_CLUSTER_HEADERS_TO_APPEND
OVERALL_HEADERS = rmc_core.OVERALL_HEADERS + [
    key for key in _STRICT_CLUSTER_HEADERS_TO_APPEND if key not in rmc_core.OVERALL_HEADERS
]
DIRECT_CLASSIFICATION_SUMMARY_HEADERS = rmc_core.DIRECT_CLASSIFICATION_SUMMARY_HEADERS
DIRECT_CLASSIFICATION_SUMMARY_OVERALL_HEADERS = (
    rmc_core.DIRECT_CLASSIFICATION_SUMMARY_OVERALL_HEADERS
)
InstanceRecord = commercial_metrics.InstanceRecord
PairRecord = commercial_metrics.PairRecord


def normalize_label(raw: Optional[str]) -> Optional[str]:
    return commercial_metrics.normalize_label(raw)


def infer_true_label_from_record(rec: Dict, file_name: str) -> Optional[str]:
    expected = normalize_label(rec.get("expected_result"))
    if expected:
        return expected
    return commercial_metrics.infer_true_label_from_filename(file_name)


def extract_pred_from_record(rec: Dict) -> Optional[str]:
    parsed = rec.get("parsed")
    if isinstance(parsed, dict):
        pred = normalize_label(parsed.get("result"))
        if pred:
            return pred

    parsed_response = rec.get("parsed_response")
    if isinstance(parsed_response, dict):
        pred = normalize_label(parsed_response.get("result"))
        if pred:
            return pred

    raw = rec.get("model_direct_output")
    if isinstance(raw, str):
        match = re.search(r'"result"\s*:\s*"([^"]+)"', raw, flags=re.IGNORECASE)
        if match:
            pred = normalize_label(match.group(1))
            if pred:
                return pred
        upper = raw.upper()
        if UNSAT in upper or "UNSAT" in upper:
            return UNSAT
        if SAT in upper or "SAT" in upper:
            return SAT

    return commercial_metrics.extract_pred_from_record(rec)


def extract_assignment_from_record(rec: Dict) -> Optional[Dict]:
    parsed = rec.get("parsed")
    if isinstance(parsed, dict) and isinstance(parsed.get("assignment"), dict):
        return parsed.get("assignment")

    raw = rec.get("model_direct_output")
    if isinstance(raw, str):
        match = re.search(
            r'(\{\s*"result"\s*:\s*"(?:SATISFIABLE|UNSATISFIABLE)".*\})\s*$',
            raw,
            flags=re.IGNORECASE | re.DOTALL,
        )
        if match:
            try:
                obj = json.loads(match.group(1))
            except Exception:
                obj = None
            if isinstance(obj, dict) and isinstance(obj.get("assignment"), dict):
                return obj.get("assignment")

    return commercial_metrics.extract_assignment_from_record(rec)


def iter_results_files(open_source_root: str) -> Iterable[str]:
    for walk_root, _, files in os.walk(open_source_root):
        if "results_desc.jsonl" in files:
            yield os.path.join(walk_root, "results_desc.jsonl")


def parse_result_line(
    results_path: str, line_no: int, rec: Dict
) -> Tuple[Optional[InstanceRecord], Optional[str]]:
    source_file = rec.get("cnf_file") if isinstance(rec.get("cnf_file"), str) else None
    prompt_file = rec.get("prompt_file") if isinstance(rec.get("prompt_file"), str) else None
    parse_path = source_file or prompt_file
    if not parse_path:
        return None, "missing_source_file"

    normalized_path = parse_path.replace("\\", "/")
    match = re.search(r"/N_(\d+)/(?:cases|cases_prompt)/([^/]+)/([^/]+)$", normalized_path)
    if not match:
        return None, "path_too_short"

    n_value = int(match.group(1))
    group_id = match.group(2)
    file_name = match.group(3)
    pair_id = commercial_metrics.infer_pair_id_from_filename(file_name)
    if not pair_id:
        return None, "invalid_pair_id"

    true_label = infer_true_label_from_record(rec, file_name)
    if true_label is None:
        return None, "invalid_true_label"

    pred = extract_pred_from_record(rec)
    assignment = extract_assignment_from_record(rec)
    json_path = "{}:{}".format(results_path, line_no)
    return (
        InstanceRecord(
            n_value, group_id, pair_id, true_label, pred, json_path, source_file, assignment
        ),
        None,
    )


def canonical_instance_dedup_key(item: InstanceRecord):
    path = item.source_file or item.json_path
    normalized = path.replace("\\", "/")
    marker = "/" + MIN_EDIT_DATASET_DIR + "/"
    if marker in normalized:
        rel = normalized.split(marker, 1)[1]
        if rel.endswith(".json"):
            rel = rel[:-5]
        return ("dataset_path", rel)
    return ("n_group_pair_label", item.n_value, item.group_id, item.pair_id, item.true_label)


def load_instances_from_results(results_path: str) -> Tuple[List[InstanceRecord], Dict[str, int]]:
    stats = {
        "total_jsonl_records_seen": 0,
        "accepted_instances": 0,
        "dropped_json_read_error": 0,
        "dropped_missing_source_file": 0,
        "dropped_path_too_short": 0,
        "dropped_invalid_pair_id": 0,
        "dropped_invalid_true_label": 0,
        "dropped_duplicate_source_file": 0,
        "missing_prediction_label": 0,
    }
    latest_by_key: Dict[Tuple, InstanceRecord] = {}

    with open(results_path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            stats["total_jsonl_records_seen"] += 1
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                stats["dropped_json_read_error"] += 1
                continue
            if not isinstance(rec, dict):
                stats["dropped_json_read_error"] += 1
                continue

            item, err = parse_result_line(results_path, line_no, rec)
            if err:
                stats["dropped_" + err] += 1
                continue
            if item.pred_label is None:
                stats["missing_prediction_label"] += 1

            key = canonical_instance_dedup_key(item)
            if key in latest_by_key:
                stats["dropped_duplicate_source_file"] += 1
            latest_by_key[key] = item

    instances = list(latest_by_key.values())
    stats["accepted_instances"] = len(instances)
    return instances, stats


def read_first_model_name(results_path: str) -> Optional[str]:
    try:
        with open(results_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                model = rec.get("model")
                if isinstance(model, str) and model:
                    return model
    except OSError:
        return None
    return None


def sanitize_model_label(raw: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "__", raw).strip("_")


def infer_model_label(open_source_root: str, results_path: str) -> str:
    output_dir = os.path.dirname(results_path)
    rel = os.path.relpath(output_dir, open_source_root).replace("\\", "/")
    parts = rel.split("/")
    if parts and parts[0] == "outputs":
        provider = os.path.basename(open_source_root.rstrip(os.sep)) or "open_source"
    elif parts and parts[0] not in {".", ""}:
        provider = parts[0]
    elif os.path.basename(open_source_root.rstrip(os.sep)) == "outputs":
        provider = os.path.basename(os.path.dirname(open_source_root.rstrip(os.sep))) or "open_source"
    else:
        provider = os.path.basename(open_source_root.rstrip(os.sep)) or "open_source"
    model = read_first_model_name(results_path) or os.path.basename(output_dir)
    return sanitize_model_label("{}__{}".format(provider, model))


def find_open_source_result_groups(open_source_root: str) -> List[Tuple[str, str]]:
    if not os.path.isdir(open_source_root):
        return []
    candidates = []
    for results_path in sorted(iter_results_files(open_source_root)):
        candidates.append((infer_model_label(open_source_root, results_path), results_path))

    label_counts: Dict[str, int] = defaultdict(int)
    for label, _ in candidates:
        label_counts[label] += 1

    used_labels = set()
    out = []
    for label, results_path in candidates:
        final_label = label
        if label_counts[label] > 1:
            output_dir_label = sanitize_model_label(os.path.basename(os.path.dirname(results_path)))
            final_label = "{}__{}".format(label, output_dir_label)
        if final_label in used_labels:
            suffix = 2
            unique_label = "{}__{}".format(final_label, suffix)
            while unique_label in used_labels:
                suffix += 1
                unique_label = "{}__{}".format(final_label, suffix)
            final_label = unique_label
        used_labels.add(final_label)
        out.append((final_label, results_path))
    return out


def parse_dimacs(cnf_text: str) -> Tuple[int, List[List[int]]]:
    n_vars: Optional[int] = None
    clauses: List[List[int]] = []
    current: List[int] = []
    for raw in cnf_text.splitlines():
        line = raw.strip()
        if not line or line.startswith("c"):
            continue
        if line.startswith("p "):
            parts = line.split()
            if len(parts) >= 4 and parts[1] == "cnf":
                n_vars = int(parts[2])
            continue
        for token in line.split():
            lit = int(token)
            if lit == 0:
                if current:
                    clauses.append(current)
                    current = []
            else:
                current.append(lit)
    if current:
        clauses.append(current)
    if n_vars is None:
        n_vars = max((abs(lit) for clause in clauses for lit in clause), default=0)
    return n_vars, clauses


def load_graph_meta(cnf_path: Path) -> Optional[Dict[str, Any]]:
    json_path = cnf_path.with_suffix(".json")
    if not json_path.exists():
        return None
    try:
        return json.loads(json_path.read_text(encoding="utf-8"))
    except Exception:
        return None


def infer_graph_from_cnf(n_vars: int, clauses: List[List[int]]) -> Dict[str, Any]:
    k_colors = 3
    for clause in clauses:
        if clause and all(lit > 0 for lit in clause) and len(clause) >= 2:
            k_colors = len(clause)
            break
    n_vertices = n_vars // k_colors if k_colors else 0
    edges = set()
    for clause in clauses:
        if len(clause) == 2 and clause[0] < 0 and clause[1] < 0:
            left = (-clause[0] - 1) // k_colors + 1
            right = (-clause[1] - 1) // k_colors + 1
            if left != right:
                edges.add((min(left, right), max(left, right)))
    return {
        "n_vertices": n_vertices,
        "k_colors": k_colors,
        "n_edges": len(edges),
        "edges": [list(edge) for edge in sorted(edges)],
    }


def resolve_source_file(path: Optional[str]) -> Optional[str]:
    if not path:
        return None
    candidates = [path]
    if path.startswith("/work/"):
        candidates.append("/ddnB" + path)
    marker = "Graph_Colouring_Evaluation/"
    if marker in path:
        rel = path.split(marker, 1)[1]
        candidates.append(os.path.join(GRAPH_ROOT, rel))
    for candidate in candidates:
        if candidate and os.path.isfile(candidate):
            return candidate
    return None


def parse_vertex_assignment(
    assignment: Dict[str, Any], n_vertices: int, k_colors: int
) -> Tuple[Optional[Dict[int, int]], str]:
    colors: Dict[int, int] = {}
    for key, value in assignment.items():
        match = re.fullmatch(r"v?(\d+)", str(key).strip(), flags=re.I)
        if not match:
            return None, "invalid vertex key: {}".format(key)
        vertex = int(match.group(1))
        if vertex < 1 or vertex > n_vertices:
            return None, "vertex out of range: {}".format(vertex)
        if isinstance(value, int):
            color = value
        elif isinstance(value, str) and value.strip().isdigit():
            color = int(value.strip())
        else:
            return None, "invalid color for vertex {}: {}".format(vertex, value)
        if color < 1 or color > k_colors:
            return None, "color out of range for vertex {}: {}".format(vertex, color)
        colors[vertex] = color
    missing = [vertex for vertex in range(1, n_vertices + 1) if vertex not in colors]
    if missing:
        return None, "assignment missing vertices: {}".format(missing[:10])
    return colors, "ok"


def validate_coloring_assignment(
    item: InstanceRecord, graph_meta: Dict[str, Any], cnf_text: str
) -> Tuple[bool, str]:
    if item.true_label == UNSAT:
        return False, "expected UNSAT, got SAT"

    assignment = item.assignment
    if not isinstance(assignment, dict):
        return False, "SAT answer missing assignment object"

    n_vertices = int(graph_meta["n_vertices"])
    k_colors = int(graph_meta["k_colors"])
    colors, message = parse_vertex_assignment(assignment, n_vertices, k_colors)
    if colors is None:
        return False, message

    for left, right in graph_meta.get("edges", []):
        if colors[int(left)] == colors[int(right)]:
            return False, "edge ({}, {}) has the same color {}".format(
                left, right, colors[int(left)]
            )

    n_vars, clauses = parse_dimacs(cnf_text)
    var_values: Dict[int, bool] = {}
    for vertex in range(1, n_vertices + 1):
        chosen = colors[vertex]
        for color in range(1, k_colors + 1):
            var = (vertex - 1) * k_colors + color
            var_values[var] = color == chosen
    if n_vars and len(var_values) < n_vars:
        return False, "assignment does not cover all encoded variables: {}/{}".format(
            len(var_values), n_vars
        )

    for idx, clause in enumerate(clauses, start=1):
        satisfied = False
        for lit in clause:
            value = var_values.get(abs(lit))
            if value is None:
                return False, "missing value for variable {}".format(abs(lit))
            if (lit > 0 and value) or (lit < 0 and not value):
                satisfied = True
                break
        if not satisfied:
            return False, "assignment does not satisfy clause {}".format(idx)
    return True, "SAT assignment is a valid graph coloring"


def verify_assignment_for_instance(item: InstanceRecord) -> str:
    source_path = resolve_source_file(item.source_file)
    if source_path is None:
        return "missing_cnf"

    if item.assignment is None:
        return "missing_assignment"
    if not isinstance(item.assignment, dict):
        return "invalid_assignment"

    try:
        cnf_text = Path(source_path).read_text(encoding="utf-8")
    except Exception:
        return "missing_cnf"

    graph_meta = load_graph_meta(Path(source_path))
    if graph_meta is None:
        try:
            n_vars, clauses = parse_dimacs(cnf_text)
            graph_meta = infer_graph_from_cnf(n_vars, clauses)
        except Exception:
            return "missing_cnf"

    valid, message = validate_coloring_assignment(item, graph_meta, cnf_text)
    if valid:
        return "satisfied"
    if "missing assignment" in message.lower():
        return "missing_assignment"
    if "missing_cnf" in message.lower():
        return "missing_cnf"
    return "unsatisfied" if item.true_label == SAT else "invalid_assignment"


def build_pairs(instances: List[InstanceRecord]) -> Tuple[List[PairRecord], Dict[str, int]]:
    bucket = defaultdict(
        lambda: {"sat_pred": None, "unsat_pred": None, "sat_assignment_satisfied": False}
    )
    stats = {"duplicate_sat_instances": 0, "duplicate_unsat_instances": 0}
    for item in instances:
        key = (item.n_value, item.group_id, item.pair_id)
        if item.true_label == SAT:
            if bucket[key]["sat_pred"] is not None:
                stats["duplicate_sat_instances"] += 1
            bucket[key]["sat_pred"] = item.pred_label
            if item.assignment is not None:
                bucket[key]["sat_assignment_satisfied"] = (
                    verify_assignment_for_instance(item) == "satisfied"
                )
        elif item.true_label == UNSAT:
            if bucket[key]["unsat_pred"] is not None:
                stats["duplicate_unsat_instances"] += 1
            bucket[key]["unsat_pred"] = item.pred_label

    pairs = []
    for (n_value, group_id, pair_id), value in bucket.items():
        pairs.append(
            PairRecord(
                n_value,
                group_id,
                pair_id,
                value["sat_pred"],
                value["unsat_pred"],
                value["sat_assignment_satisfied"],
            )
        )
    stats["pairs_total"] = len(pairs)
    return pairs, stats


def _safe_div(num: float, den: float) -> Optional[float]:
    return commercial_metrics._safe_div(num, den)


def _binary_stats(instances: List[InstanceRecord], positive_label: str) -> Dict[str, int]:
    return commercial_metrics._binary_stats(instances, positive_label)


def compute_assignment_metrics(instances: List[InstanceRecord]) -> Dict:
    pred_sat = [x for x in instances if x.pred_label == SAT]
    pred_unsat = [x for x in instances if x.pred_label == UNSAT]
    counts = {
        "assignment_num_pred_sat": len(pred_sat),
        "assignment_num_with_assignment": 0,
        "assignment_num_complete_assignment": 0,
        "assignment_num_checked": 0,
        "assignment_num_satisfied": 0,
        "assignment_num_unsatisfied": 0,
        "assignment_num_missing_assignment": 0,
        "assignment_num_invalid_assignment": 0,
        "assignment_num_missing_cnf": 0,
        "assignment_num_cdcl_unavailable": 0,
    }

    for item in pred_sat:
        if item.assignment is not None:
            counts["assignment_num_with_assignment"] += 1
        status = verify_assignment_for_instance(item)
        if status == "satisfied":
            counts["assignment_num_complete_assignment"] += 1
            counts["assignment_num_checked"] += 1
            counts["assignment_num_satisfied"] += 1
        elif status == "unsatisfied":
            counts["assignment_num_complete_assignment"] += 1
            counts["assignment_num_checked"] += 1
            counts["assignment_num_unsatisfied"] += 1
        elif status == "missing_assignment":
            counts["assignment_num_missing_assignment"] += 1
        elif status == "invalid_assignment":
            counts["assignment_num_invalid_assignment"] += 1
        elif status == "missing_cnf":
            counts["assignment_num_missing_cnf"] += 1
        elif status == "cdcl_unavailable":
            counts["assignment_num_complete_assignment"] += 1
            counts["assignment_num_cdcl_unavailable"] += 1

    counts["assignment_accuracy"] = _safe_div(
        counts["assignment_num_satisfied"], counts["assignment_num_pred_sat"]
    )
    counts["assignment_checked_accuracy"] = _safe_div(
        counts["assignment_num_satisfied"], counts["assignment_num_checked"]
    )
    counts["assignment_or_unsat_accuracy"] = _safe_div(
        counts["assignment_num_satisfied"] + sum(1 for item in pred_unsat if item.true_label == UNSAT),
        len(pred_sat) + len(pred_unsat),
    )
    return counts


def compute_instance_metrics(instances: List[InstanceRecord]) -> Dict:
    valid = [x for x in instances if x.pred_label in {SAT, UNSAT}]
    total = len(valid)
    total_all = len(instances)
    sat_stats = _binary_stats(valid, SAT)
    unsat_stats = _binary_stats(valid, UNSAT)

    num_true_sat_all = sum(1 for x in instances if x.true_label == SAT)
    num_true_unsat_all = sum(1 for x in instances if x.true_label == UNSAT)
    num_true_sat_eval = sum(1 for x in valid if x.true_label == SAT)
    num_true_unsat_eval = sum(1 for x in valid if x.true_label == UNSAT)
    num_missing_prediction = total_all - total

    accuracy = _safe_div(sat_stats["tp"] + sat_stats["tn"], total)
    precision_sat = _safe_div(sat_stats["tp"], sat_stats["tp"] + sat_stats["fp"])
    recall_sat = _safe_div(sat_stats["tp"], sat_stats["tp"] + sat_stats["fn"])
    f1_sat = (
        _safe_div(2 * precision_sat * recall_sat, precision_sat + recall_sat)
        if precision_sat is not None and recall_sat is not None and (precision_sat + recall_sat) > 0
        else None
    )

    precision_unsat = _safe_div(unsat_stats["tp"], unsat_stats["tp"] + unsat_stats["fp"])
    recall_unsat = _safe_div(unsat_stats["tp"], unsat_stats["tp"] + unsat_stats["fn"])
    f1_unsat = (
        _safe_div(2 * precision_unsat * recall_unsat, precision_unsat + recall_unsat)
        if precision_unsat is not None
        and recall_unsat is not None
        and (precision_unsat + recall_unsat) > 0
        else None
    )

    tp = sat_stats["tp"]
    tn = sat_stats["tn"]
    fp = sat_stats["fp"]
    fn = sat_stats["fn"]
    mcc_den = ((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) ** 0.5
    mcc = _safe_div((tp * tn - fp * fn), mcc_den) if mcc_den != 0 else None

    metrics = {
        "num_instances_all": total_all,
        "num_instances_eval": total,
        "num_missing_prediction": num_missing_prediction,
        "num_true_sat_all": num_true_sat_all,
        "num_true_unsat_all": num_true_unsat_all,
        "num_true_sat_eval": num_true_sat_eval,
        "num_true_unsat_eval": num_true_unsat_eval,
        "num_true_sat_pred_sat": sat_stats["tp"],
        "num_true_sat_pred_unsat": sat_stats["fn"],
        "num_true_unsat_pred_sat": sat_stats["fp"],
        "num_true_unsat_pred_unsat": sat_stats["tn"],
        "ratio_true_sat_pred_sat": _safe_div(sat_stats["tp"], num_true_sat_eval),
        "ratio_true_sat_pred_unsat": _safe_div(sat_stats["fn"], num_true_sat_eval),
        "ratio_true_unsat_pred_sat": _safe_div(sat_stats["fp"], num_true_unsat_eval),
        "ratio_true_unsat_pred_unsat": _safe_div(sat_stats["tn"], num_true_unsat_eval),
        "instance_accuracy": accuracy,
        "instance_mcc": mcc,
        "sat_positive_precision": precision_sat,
        "sat_positive_recall": recall_sat,
        "sat_positive_f1": f1_sat,
        "unsat_positive_precision": precision_unsat,
        "unsat_positive_recall": recall_unsat,
        "unsat_positive_f1": f1_unsat,
    }
    metrics.update(compute_assignment_metrics(instances))
    return rmc_core.add_derived_adr_metrics(metrics)


def compute_strict_cluster_metrics(pairs: List[PairRecord], adr: Optional[float]) -> Dict:
    grouped = defaultdict(list)
    for pair in pairs:
        grouped[(pair.n_value, pair.group_id)].append(pair)

    cluster_rows = []
    for (n_value, group_id) in sorted(grouped):
        plist = grouped[(n_value, group_id)]
        total_pairs = len(plist)
        complete_pairs = 0
        success_pairs = 0
        for pair in plist:
            pair_complete = pair.sat_pred is not None and pair.unsat_pred is not None
            if pair_complete:
                complete_pairs += 1
                if pair.sat_pred == SAT and pair.unsat_pred == UNSAT:
                    success_pairs += 1

        all_pairs_complete = total_pairs > 0 and complete_pairs == total_pairs
        all_pairs_success = all_pairs_complete and success_pairs == total_pairs
        cluster_rows.append(
            {
                "N": n_value,
                "group_id": group_id,
                "num_pairs": total_pairs,
                "num_complete_pairs": complete_pairs,
                "num_success_pairs": success_pairs,
                "all_pairs_complete": all_pairs_complete,
                "all_pairs_success": all_pairs_success,
            }
        )

    num_clusters = len(cluster_rows)
    num_complete = sum(1 for row in cluster_rows if row["all_pairs_complete"])
    num_success = sum(1 for row in cluster_rows if row["all_pairs_success"])
    strict_success_rate = _safe_div(num_success, num_clusters)
    return {
        "num_clusters_total": num_clusters,
        "num_clusters_all_pairs_complete": num_complete,
        "num_clusters_all_pairs_success": num_success,
        "cluster_all_pairs_complete_rate": _safe_div(num_complete, num_clusters),
        "cluster_all_pairs_success_rate": strict_success_rate,
        "cluster_success_gap_ADR_minus_strict_cluster_success": (
            adr - strict_success_rate if adr is not None and strict_success_rate is not None else None
        ),
        "strict_cluster_rows": cluster_rows,
    }


def row_from_metrics(model: str, n_value: int, metrics: Dict) -> Dict:
    row = {"model": model, "N": n_value}
    for key in BY_N_HEADERS[2:]:
        row[key] = metrics.get(key)
    return row


def overall_row_from_metrics(model: str, input_root: str, metrics: Dict) -> Dict:
    row = {"model": model, "input_root": input_root}
    for key in OVERALL_HEADERS[2:]:
        row[key] = metrics.get(key)
    return row


def write_csv(path: str, headers: List[str], rows: List[Dict]) -> None:
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
