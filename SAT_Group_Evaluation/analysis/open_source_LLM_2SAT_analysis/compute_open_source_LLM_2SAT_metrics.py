#!/usr/bin/env python3
"""Compute metrics for open-source LLM 2SAT prediction outputs.

This mirrors analysis/commercial_api_results_analysis/compute_commercial_api_metrics.py,
but reads each open-source model's results_desc.jsonl file under open_source_LLM_2SAT.
"""

import argparse
import csv
import json
import os
import re
import sys
from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Tuple


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ANALYSIS_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, os.pardir))
REPO_ROOT = os.path.abspath(os.path.join(ANALYSIS_DIR, os.pardir))
COMMERCIAL_ANALYSIS_DIR = os.path.join(ANALYSIS_DIR, "commercial_api_results_analysis")
if COMMERCIAL_ANALYSIS_DIR not in sys.path:
    sys.path.insert(0, COMMERCIAL_ANALYSIS_DIR)

import compute_commercial_api_metrics as commercial_metrics  # noqa: E402


SAT = commercial_metrics.SAT
UNSAT = commercial_metrics.UNSAT
DEFAULT_OPEN_SOURCE_ROOT = os.path.join(REPO_ROOT, "open_source_LLM_2SAT")
DEFAULT_OUTPUT_DIR = SCRIPT_DIR
MIN_EDIT_DATASET_DIR = "min_edit_2sat_dataset"
PREFERRED_PROVIDER_DIRS = (
    "qwen",
    "deepseek",
    "gemma",
    "llama3_1",
    "OpenAI",
    "NVIDIA",
)

commercial_metrics.MIN_EDIT_DATASET_DIR = MIN_EDIT_DATASET_DIR

UNSAT_CLAUSE_SUBSET_PAIR_HEADERS = [
    "pair_unsat_clause_subset_available_total",
    "pair_unsat_clause_subset_verified_total",
    "pair_success_with_assignment_and_unsat_subset_total",
    "GADR_with_assignment_and_unsat_subset",
    "ADR_with_assignment_and_unsat_subset",
    "SGADR_with_assignment_and_unsat_subset",
    "SGADR_star_with_assignment_and_unsat_subset",
]

UNSAT_CLAUSE_SUBSET_INSTANCE_HEADERS = [
    "unsat_clause_subset_num_pred_unsat",
    "unsat_clause_subset_num_with_followup_result",
    "unsat_clause_subset_num_success",
    "unsat_clause_subset_num_failed",
    "unsat_clause_subset_num_missing_followup",
    "unsat_clause_subset_coverage",
    "unsat_clause_subset_success_rate",
    "unsat_clause_subset_verified_rate",
]

STRICT_CLUSTER_HEADERS = [
    "num_clusters_total",
    "num_clusters_all_pairs_complete",
    "num_clusters_all_pairs_success",
    "cluster_all_pairs_complete_rate",
    "cluster_all_pairs_success_rate",
    "cluster_success_gap_ADR_minus_strict_cluster_success",
]

_STRICT_CLUSTER_HEADERS_TO_APPEND = [
    key for key in STRICT_CLUSTER_HEADERS if key not in commercial_metrics.BY_N_HEADERS
]
_UNSAT_SUBSET_HEADERS_TO_APPEND = [
    key
    for key in (UNSAT_CLAUSE_SUBSET_PAIR_HEADERS + UNSAT_CLAUSE_SUBSET_INSTANCE_HEADERS)
    if key not in commercial_metrics.BY_N_HEADERS
]
BY_N_HEADERS = (
    commercial_metrics.BY_N_HEADERS
    + _UNSAT_SUBSET_HEADERS_TO_APPEND
    + _STRICT_CLUSTER_HEADERS_TO_APPEND
)
OVERALL_HEADERS = commercial_metrics.OVERALL_HEADERS + [
    key
    for key in (_UNSAT_SUBSET_HEADERS_TO_APPEND + _STRICT_CLUSTER_HEADERS_TO_APPEND)
    if key not in commercial_metrics.OVERALL_HEADERS
]
InstanceRecord = commercial_metrics.InstanceRecord


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
    return commercial_metrics.extract_assignment_from_record(rec)


def iter_results_files(open_source_root: str) -> Iterable[str]:
    seen = set()

    def emit_from(root: str):
        for walk_root, _, files in os.walk(root):
            if "results_desc.jsonl" not in files:
                continue
            path = os.path.join(walk_root, "results_desc.jsonl")
            if path in seen:
                continue
            seen.add(path)
            yield path

    for provider in PREFERRED_PROVIDER_DIRS:
        provider_root = os.path.join(open_source_root, provider)
        if os.path.isdir(provider_root):
            yield from emit_from(provider_root)

    yield from emit_from(open_source_root)


def parse_result_line(results_path: str, line_no: int, rec: Dict) -> Tuple[Optional[InstanceRecord], Optional[str]]:
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
    return InstanceRecord(n_value, group_id, pair_id, true_label, pred, json_path, source_file, assignment), None


def canonical_instance_dedup_key(item: InstanceRecord):
    path = item.source_file or item.json_path
    rel = canonical_dataset_relative_path(path)
    if rel is not None:
        return ("dataset_path", rel)
    return ("n_group_pair_label", item.n_value, item.group_id, item.pair_id, item.true_label)


def canonical_dataset_relative_path(path: Optional[str]) -> Optional[str]:
    if not path:
        return None
    normalized = path.replace("\\", "/")
    marker = "/" + MIN_EDIT_DATASET_DIR + "/"
    if marker in normalized:
        rel = normalized.split(marker, 1)[1]
        if rel.endswith(".json"):
            rel = rel[:-5]
        return rel
    return None


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


def normalize_output_dir_signature(name: str) -> str:
    signature = str(name)
    patterns = [
        r"_cases_prompt_interleave_zero_shot_desc",
        r"_cases_prompt_interleave_desc",
        r"_cases_prompt_interleave_zero_shot",
        r"_cases_prompt_interleave",
        r"_unsat_clause_subset_followup",
    ]
    for pattern in patterns:
        updated = re.sub(pattern, "_TASK", signature, count=1)
        if updated != signature:
            return updated
    return signature


def find_matching_unsat_clause_subset_results(
    open_source_root: str,
    base_results_path: str,
) -> Optional[str]:
    output_dir = os.path.dirname(base_results_path)
    output_name = os.path.basename(output_dir)
    if "unsat_clause_subset_followup" in output_name:
        return None

    provider_dir = os.path.dirname(os.path.dirname(output_dir))
    outputs_dir = os.path.join(provider_dir, "outputs")
    if not os.path.isdir(outputs_dir):
        return None

    base_signature = normalize_output_dir_signature(output_name)
    candidates: List[str] = []
    for walk_root, _, files in os.walk(outputs_dir):
        if "results_desc.jsonl" not in files:
            continue
        candidate_dir = os.path.basename(walk_root)
        if "unsat_clause_subset_followup" not in candidate_dir:
            continue
        if normalize_output_dir_signature(candidate_dir) == base_signature:
            candidates.append(os.path.join(walk_root, "results_desc.jsonl"))

    if not candidates:
        return None
    candidates.sort(
        key=lambda path: (
            os.path.basename(os.path.dirname(path)) != output_name,
            path,
        )
    )
    return candidates[0]


def load_unsat_clause_subset_results(results_path: str) -> Tuple[Dict[Tuple[str, str], bool], Dict[str, int]]:
    stats = {
        "total_jsonl_records_seen": 0,
        "accepted_followup_records": 0,
        "dropped_json_read_error": 0,
        "dropped_missing_cnf_file": 0,
        "dropped_path_outside_dataset": 0,
        "dropped_duplicate_cnf_file": 0,
        "followup_success_records": 0,
        "followup_failed_records": 0,
    }
    latest_by_key: Dict[Tuple[str, str], bool] = {}

    with open(results_path, "r", encoding="utf-8") as f:
        for line in f:
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

            cnf_file = rec.get("cnf_file")
            if not isinstance(cnf_file, str) or not cnf_file:
                stats["dropped_missing_cnf_file"] += 1
                continue
            rel = canonical_dataset_relative_path(cnf_file)
            if rel is None:
                stats["dropped_path_outside_dataset"] += 1
                continue

            success = bool(rec.get("success"))
            key = ("dataset_path", rel)
            if key in latest_by_key:
                stats["dropped_duplicate_cnf_file"] += 1
            latest_by_key[key] = success

    for success in latest_by_key.values():
        if success:
            stats["followup_success_records"] += 1
        else:
            stats["followup_failed_records"] += 1
    stats["accepted_followup_records"] = len(latest_by_key)
    return latest_by_key, stats


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


def _safe_div(num: float, den: float) -> Optional[float]:
    return commercial_metrics._safe_div(num, den)


def compute_strict_cluster_metrics(pairs: List[commercial_metrics.PairRecord], adr: Optional[float]) -> Dict:
    """Compute cluster-level all-variants success without changing existing ADR logic.

    A cluster is keyed by (N, group_id). It succeeds only when every pair in the
    cluster is complete and every pair is accurately differentiated.
    """
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


def build_open_source_pairs(
    instances: List[InstanceRecord],
    unsat_subset_by_key: Optional[Dict[Tuple[str, str], bool]] = None,
) -> Tuple[List[commercial_metrics.PairRecord], Dict[str, int]]:
    if unsat_subset_by_key is None:
        unsat_subset_by_key = {}

    bucket = defaultdict(
        lambda: {
            "sat_pred": None,
            "unsat_pred": None,
            "sat_assignment_satisfied": False,
            "unsat_subset_verified": None,
        }
    )
    stats = {
        "pairs_total": 0,
        "duplicate_conflict_sat": 0,
        "duplicate_conflict_unsat": 0,
        "duplicate_conflict_unsat_subset": 0,
    }

    for item in instances:
        key = (item.n_value, item.group_id, item.pair_id)
        slot = "sat_pred" if item.true_label == SAT else "unsat_pred"
        old = bucket[key][slot]
        if old is not None and item.pred_label is not None and old != item.pred_label:
            stats["duplicate_conflict_sat" if slot == "sat_pred" else "duplicate_conflict_unsat"] += 1
        if item.pred_label is not None:
            bucket[key][slot] = item.pred_label
            if item.true_label == SAT and item.pred_label == SAT:
                bucket[key]["sat_assignment_satisfied"] = (
                    commercial_metrics.verify_assignment_for_instance(item) == "satisfied"
                )
            elif item.true_label == UNSAT and item.pred_label == UNSAT:
                subset_key = canonical_instance_dedup_key(item)
                subset_value = unsat_subset_by_key.get(subset_key)
                old_subset = bucket[key]["unsat_subset_verified"]
                if (
                    old_subset is not None
                    and subset_value is not None
                    and old_subset != subset_value
                ):
                    stats["duplicate_conflict_unsat_subset"] += 1
                if subset_value is not None:
                    bucket[key]["unsat_subset_verified"] = bool(subset_value)

    pairs = []
    for (n_value, group_id, pair_id), value in bucket.items():
        pair = commercial_metrics.PairRecord(
            n_value,
            group_id,
            pair_id,
            value["sat_pred"],
            value["unsat_pred"],
            value["sat_assignment_satisfied"],
        )
        pair.unsat_subset_verified = value["unsat_subset_verified"]
        pairs.append(pair)
    stats["pairs_total"] = len(pairs)
    return pairs, stats


def compute_unsat_clause_subset_instance_metrics(
    instances: List[InstanceRecord],
    unsat_subset_by_key: Optional[Dict[Tuple[str, str], bool]] = None,
) -> Dict:
    if unsat_subset_by_key is None:
        unsat_subset_by_key = {}

    pred_unsat = [x for x in instances if x.pred_label == UNSAT]
    with_followup = 0
    success = 0
    failed = 0
    for item in pred_unsat:
        subset_key = canonical_instance_dedup_key(item)
        subset_value = unsat_subset_by_key.get(subset_key)
        if subset_value is None:
            continue
        with_followup += 1
        if subset_value:
            success += 1
        else:
            failed += 1

    total_pred_unsat = len(pred_unsat)
    missing_followup = total_pred_unsat - with_followup
    return {
        "unsat_clause_subset_num_pred_unsat": total_pred_unsat,
        "unsat_clause_subset_num_with_followup_result": with_followup,
        "unsat_clause_subset_num_success": success,
        "unsat_clause_subset_num_failed": failed,
        "unsat_clause_subset_num_missing_followup": missing_followup,
        "unsat_clause_subset_coverage": _safe_div(with_followup, total_pred_unsat),
        "unsat_clause_subset_success_rate": _safe_div(success, with_followup),
        "unsat_clause_subset_verified_rate": _safe_div(success, total_pred_unsat),
    }


def augment_pair_metrics_with_unsat_clause_subset(
    pairs: List[commercial_metrics.PairRecord],
    metrics: Dict,
    sgadr_lambda: float = 1.0,
) -> Dict:
    grouped = defaultdict(list)
    for pair in pairs:
        grouped[pair.group_id].append(pair)

    total_available = 0
    total_verified = 0
    total_success = 0
    valid_group_rows = []
    for gid in sorted(grouped):
        plist = grouped[gid]
        complete = 0
        available = 0
        verified = 0
        success = 0
        for pair in plist:
            if pair.sat_pred is None or pair.unsat_pred is None:
                continue
            complete += 1
            subset_value = getattr(pair, "unsat_subset_verified", None)
            subset_available = subset_value is not None
            subset_verified = bool(subset_value)
            pair_success = int(
                pair.sat_pred == SAT
                and pair.unsat_pred == UNSAT
                and pair.sat_assignment_satisfied
                and subset_verified
            )
            available += int(subset_available)
            verified += int(subset_verified)
            success += pair_success

        if complete <= 0:
            continue

        group_adr = _safe_div(success, complete)
        group_variance = group_adr * (1.0 - group_adr) if group_adr is not None else None
        group_sgadr = (
            group_adr - sgadr_lambda * group_variance
            if group_adr is not None and group_variance is not None
            else None
        )
        valid_group_rows.append(
            {
                "group_id": gid,
                "pair_unsat_clause_subset_available_count": available,
                "pair_unsat_clause_subset_verified_count": verified,
                "pair_success_with_assignment_and_unsat_subset_count": success,
                "ADR_with_assignment_and_unsat_subset_g": group_adr,
                "ADR_variance_with_assignment_and_unsat_subset_g": group_variance,
                "SGADR_with_assignment_and_unsat_subset_g": group_sgadr,
            }
        )
        total_available += available
        total_verified += verified
        total_success += success

    valid_group_adrs = [
        row["ADR_with_assignment_and_unsat_subset_g"] for row in valid_group_rows
        if row["ADR_with_assignment_and_unsat_subset_g"] is not None
    ]
    valid_group_sgadrs = [
        row["SGADR_with_assignment_and_unsat_subset_g"] for row in valid_group_rows
        if row["SGADR_with_assignment_and_unsat_subset_g"] is not None
    ]
    total_complete = metrics.get("num_complete_pairs_total", 0)
    adr = _safe_div(total_success, total_complete) if total_complete > 0 else None
    sgadr_star = (
        adr - sgadr_lambda * adr * (1.0 - adr)
        if adr is not None
        else None
    )

    metrics.update(
        {
            "pair_unsat_clause_subset_available_total": total_available,
            "pair_unsat_clause_subset_verified_total": total_verified,
            "pair_success_with_assignment_and_unsat_subset_total": total_success,
            "GADR_with_assignment_and_unsat_subset": (
                _safe_div(sum(valid_group_adrs), len(valid_group_adrs)) if valid_group_adrs else None
            ),
            "ADR_with_assignment_and_unsat_subset": adr,
            "SGADR_with_assignment_and_unsat_subset": (
                _safe_div(sum(valid_group_sgadrs), len(valid_group_sgadrs)) if valid_group_sgadrs else None
            ),
            "SGADR_star_with_assignment_and_unsat_subset": sgadr_star,
        }
    )

    group_rows = metrics.get("groups")
    if isinstance(group_rows, list):
        extra_by_gid = {row["group_id"]: row for row in valid_group_rows}
        for row in group_rows:
            extra = extra_by_gid.get(row.get("group_id"))
            if extra is None:
                row.setdefault("pair_unsat_clause_subset_available_count", 0)
                row.setdefault("pair_unsat_clause_subset_verified_count", 0)
                row.setdefault("pair_success_with_assignment_and_unsat_subset_count", 0)
                row.setdefault("ADR_with_assignment_and_unsat_subset_g", None)
                row.setdefault("ADR_variance_with_assignment_and_unsat_subset_g", None)
                row.setdefault("SGADR_with_assignment_and_unsat_subset_g", None)
            else:
                row.update(extra)
    return metrics


def compute_open_source_pair_metrics(
    pairs: List[commercial_metrics.PairRecord], sgadr_lambda: float = 1.0
) -> Dict:
    metrics = commercial_metrics.compute_group_adr_metrics(pairs, sgadr_lambda=sgadr_lambda)
    augment_pair_metrics_with_unsat_clause_subset(pairs, metrics, sgadr_lambda=sgadr_lambda)
    metrics.update(compute_strict_cluster_metrics(pairs, metrics.get("ADR")))
    return metrics


def compute_open_source_by_n(
    pairs: List[commercial_metrics.PairRecord],
    instances: List[InstanceRecord],
    unsat_subset_by_key: Optional[Dict[Tuple[str, str], bool]] = None,
    sgadr_lambda: float = 1.0,
) -> List[Dict]:
    pairs_by_n = defaultdict(list)
    inst_by_n = defaultdict(list)
    for pair in pairs:
        pairs_by_n[pair.n_value].append(pair)
    for item in instances:
        inst_by_n[item.n_value].append(item)

    rows = []
    for n_value in sorted(set(list(pairs_by_n.keys()) + list(inst_by_n.keys()))):
        pair_metrics = compute_open_source_pair_metrics(pairs_by_n.get(n_value, []), sgadr_lambda=sgadr_lambda)
        inst_metrics = commercial_metrics.compute_instance_metrics(inst_by_n.get(n_value, []))
        inst_metrics.update(
            compute_unsat_clause_subset_instance_metrics(
                inst_by_n.get(n_value, []),
                unsat_subset_by_key=unsat_subset_by_key,
            )
        )
        rows.append({"N": n_value, "metrics": dict(pair_metrics, **inst_metrics)})
    return rows


def row_from_metrics(model: str, n_value: int, metrics: Dict) -> Dict:
    row = {"model": model, "N": n_value}
    for key in BY_N_HEADERS[2:]:
        row[key] = metrics[key]
    return row


def overall_row_from_metrics(model: str, input_root: str, metrics: Dict) -> Dict:
    row = {"model": model, "input_root": input_root}
    for key in OVERALL_HEADERS[2:]:
        row[key] = metrics[key]
    return row


def write_csv(path: str, headers: List[str], rows: List[Dict]) -> None:
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_excel(path: str, by_n_rows: List[Dict], overall_rows: List[Dict]) -> Tuple[bool, Optional[str]]:
    return commercial_metrics.write_excel(path, by_n_rows, overall_rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compute Group ADR, instance metrics, and assignment metrics for open-source "
            "LLM 2SAT outputs using the same metric definitions as commercial API analysis."
        )
    )
    parser.add_argument("--open-source-root", default=DEFAULT_OPEN_SOURCE_ROOT)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--preview-limit", type=int, default=40)
    parser.add_argument("--sgadr-lambda", type=float, default=1.0)
    parser.add_argument(
        "--model-filter",
        action="append",
        default=[],
        help=(
            "Run only models whose inferred label contains this substring. "
            "May be passed multiple times. Omit this option to run all models."
        ),
    )
    parser.add_argument(
        "--all-models",
        action="store_true",
        help="Run all discovered models explicitly. This is also the default when no --model-filter is passed.",
    )
    args = parser.parse_args()

    open_source_root = os.path.abspath(args.open_source_root)
    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    groups = find_open_source_result_groups(open_source_root)
    if not groups:
        raise SystemExit("No results_desc.jsonl files found under {}".format(open_source_root))
    if args.model_filter and args.all_models:
        raise SystemExit("Use either --model-filter or --all-models, not both.")
    if args.model_filter:
        filters = [x.strip() for x in args.model_filter if x and x.strip()]
        groups = [
            (model, results_path)
            for model, results_path in groups
            if any(pattern in model for pattern in filters)
        ]
        if not groups:
            raise SystemExit(
                "No models matched filters {}. Available models: {}".format(
                    filters,
                    ", ".join(model for model, _ in find_open_source_result_groups(open_source_root)),
                )
            )

    all_reports: List[Dict] = []
    by_n_rows: List[Dict] = []
    overall_rows: List[Dict] = []
    pairwise_preview_by_model: Dict[str, List[Dict]] = {}

    for model, results_path in groups:
        print("[info] loading model={} results={}".format(model, results_path))
        instances, load_stats = load_instances_from_results(results_path)
        followup_results_path = find_matching_unsat_clause_subset_results(open_source_root, results_path)
        if followup_results_path:
            unsat_subset_by_key, unsat_subset_stats = load_unsat_clause_subset_results(followup_results_path)
        else:
            unsat_subset_by_key, unsat_subset_stats = {}, {
                "total_jsonl_records_seen": 0,
                "accepted_followup_records": 0,
                "dropped_json_read_error": 0,
                "dropped_missing_cnf_file": 0,
                "dropped_path_outside_dataset": 0,
                "dropped_duplicate_cnf_file": 0,
                "followup_success_records": 0,
                "followup_failed_records": 0,
            }
        pairs, pair_stats = build_open_source_pairs(instances, unsat_subset_by_key=unsat_subset_by_key)
        overall_pair = compute_open_source_pair_metrics(pairs, sgadr_lambda=args.sgadr_lambda)
        overall_inst = commercial_metrics.compute_instance_metrics(instances)
        overall_inst.update(
            compute_unsat_clause_subset_instance_metrics(
                instances,
                unsat_subset_by_key=unsat_subset_by_key,
            )
        )
        overall = {**overall_pair, **overall_inst}
        by_n = compute_open_source_by_n(
            pairs,
            instances,
            unsat_subset_by_key=unsat_subset_by_key,
            sgadr_lambda=args.sgadr_lambda,
        )
        pairwise_preview = commercial_metrics.build_pairwise_preview(instances, args.preview_limit)

        all_reports.append(
            {
                "model": model,
                "input_root": results_path,
                "unsat_clause_subset_followup_results": followup_results_path,
                "load_stats": load_stats,
                "pair_stats": pair_stats,
                "unsat_clause_subset_stats": unsat_subset_stats,
                "overall_metrics": overall,
                "metrics_by_N": by_n,
            }
        )
        pairwise_preview_by_model[model] = pairwise_preview

        for row in by_n:
            by_n_rows.append(row_from_metrics(model, row["N"], row["metrics"]))
        overall_rows.append(overall_row_from_metrics(model, results_path, overall))

        print(
            "[info] done model={} instances={} pairs={} groups={}".format(
                model,
                load_stats["accepted_instances"],
                pair_stats["pairs_total"],
                overall["num_groups_total"],
            )
        )

    report_json = os.path.join(output_dir, "open_source_LLM_2SAT_group_adr_report.json")
    with open(report_json, "w", encoding="utf-8") as f:
        json.dump(
            {
                "open_source_root": open_source_root,
                "input_results_by_model": {model: results_path for model, results_path in groups},
                "sgadr_lambda": args.sgadr_lambda,
                "models": all_reports,
                "pairwise_preview_by_model": pairwise_preview_by_model,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    by_n_csv = os.path.join(output_dir, "open_source_LLM_2SAT_group_adr_report.by_N.csv")
    write_csv(by_n_csv, BY_N_HEADERS, sorted(by_n_rows, key=lambda x: (str(x["model"]), int(x["N"]))))

    overall_csv = os.path.join(output_dir, "open_source_LLM_2SAT_group_adr_report.overall.csv")
    write_csv(overall_csv, OVERALL_HEADERS, sorted(overall_rows, key=lambda x: str(x["model"])))

    excel_path = os.path.join(output_dir, "open_source_LLM_2SAT_group_adr_report.xlsx")
    excel_written, excel_error = write_excel(excel_path, by_n_rows, overall_rows)
    if not excel_written:
        print("[warn] excel export skipped: {}".format(excel_error))

    preview_json = os.path.join(output_dir, "open_source_LLM_2SAT_group_adr_report.pairwise_preview.json")
    with open(preview_json, "w", encoding="utf-8") as f:
        json.dump(pairwise_preview_by_model, f, ensure_ascii=False, indent=2)

    print(
        json.dumps(
            {
                "open_source_root": open_source_root,
                "num_models": len(groups),
                "models": [model for model, _ in groups],
                "output_dir": output_dir,
                "report_json": report_json,
                "by_n_csv": by_n_csv,
                "overall_csv": overall_csv,
                "excel_xlsx": excel_path if excel_written else None,
                "pairwise_preview_json": preview_json,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
