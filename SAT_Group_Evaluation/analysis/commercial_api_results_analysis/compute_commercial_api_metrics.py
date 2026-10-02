#!/usr/bin/env python3
import argparse
import csv
import json
import os
import re
from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Tuple


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ANALYSIS_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, os.pardir))
REPO_ROOT = os.path.abspath(os.path.join(ANALYSIS_DIR, os.pardir))


SAT = "SATISFIABLE"
UNSAT = "UNSATISFIABLE"
DEFAULT_COMMERCIAL_API_ROOT = os.path.join(REPO_ROOT, "commercial_api")
DEFAULT_OUTPUT_DIR = SCRIPT_DIR
MIN_EDIT_DATASET_DIR = "min_edit_3sat_dataset"
PAIR_SUCCESS_THRESHOLDS = range(1, 11)

STRICT_CLUSTER_HEADERS = [
    "num_clusters_total",
    "num_clusters_all_pairs_complete",
    "num_clusters_all_pairs_success",
    "cluster_all_pairs_complete_rate",
    "cluster_all_pairs_success_rate",
    "cluster_success_gap_ADR_minus_strict_cluster_success",
]

BY_N_HEADERS = [
    "model",
    "N",
    "num_groups_total",
    "num_groups_with_complete_pairs",
    "num_pairs_total",
    "num_complete_pairs_total",
    "pair_success_both_correct_total",
    "pair_success_with_assignment_total",
    "GADR",
    "ADR",
    "SGADR",
    "SGADR_star",
    "GADR_with_assignment",
    "ADR_with_assignment",
    "SGADR_with_assignment",
    "SGADR_star_with_assignment",
    "group_level_ADR_at_least_1_pair_success",
    "group_level_ADR_at_least_2_pair_success",
    "group_level_ADR_at_least_3_pair_success",
    "group_level_ADR_at_least_4_pair_success",
    "group_level_ADR_at_least_5_pair_success",
    "group_level_ADR_at_least_6_pair_success",
    "group_level_ADR_at_least_7_pair_success",
    "group_level_ADR_at_least_8_pair_success",
    "group_level_ADR_at_least_9_pair_success",
    "group_level_ADR_at_least_10_pair_success",
    "num_instances_all",
    "num_instances_eval",
    "num_missing_prediction",
    "num_true_sat_all",
    "num_true_unsat_all",
    "num_true_sat_eval",
    "num_true_unsat_eval",
    "num_true_sat_pred_sat",
    "num_true_sat_pred_unsat",
    "num_true_unsat_pred_sat",
    "num_true_unsat_pred_unsat",
    "ratio_true_sat_pred_sat",
    "ratio_true_sat_pred_unsat",
    "ratio_true_unsat_pred_sat",
    "ratio_true_unsat_pred_unsat",
    "instance_accuracy",
    "instance_mcc",
    "direct_separability_ability",
    "g_mean",
    "balanced_accuracy",
    "youden_j",
    "sat_positive_precision",
    "sat_positive_recall",
    "sat_positive_f1",
    "unsat_positive_precision",
    "unsat_positive_recall",
    "unsat_positive_f1",
    "assignment_num_pred_sat",
    "assignment_num_with_assignment",
    "assignment_num_complete_assignment",
    "assignment_num_checked",
    "assignment_num_satisfied",
    "assignment_num_unsatisfied",
    "assignment_num_missing_assignment",
    "assignment_num_invalid_assignment",
    "assignment_num_missing_cnf",
    "assignment_num_cdcl_unavailable",
    "assignment_accuracy",
    "assignment_checked_accuracy",
    "assignment_or_unsat_accuracy",
]
BY_N_HEADERS = BY_N_HEADERS + STRICT_CLUSTER_HEADERS

OVERALL_HEADERS = ["model", "input_root"] + BY_N_HEADERS[2:]
DIRECT_CLASSIFICATION_SUMMARY_HEADERS = [
    "model",
    "N",
    "direct_separability_ability",
    "g_mean",
    "balanced_accuracy",
    "youden_j",
    "instance_mcc",
]
DIRECT_CLASSIFICATION_SUMMARY_OVERALL_HEADERS = [
    "model",
    "input_root",
    "direct_separability_ability",
    "g_mean",
    "balanced_accuracy",
    "youden_j",
    "instance_mcc",
]


class InstanceRecord:
    def __init__(
        self,
        n_value,
        group_id,
        pair_id,
        true_label,
        pred_label,
        json_path,
        source_file,
        assignment,
    ):
        self.n_value = n_value
        self.group_id = group_id
        self.pair_id = pair_id
        self.true_label = true_label
        self.pred_label = pred_label
        self.json_path = json_path
        self.source_file = source_file
        self.assignment = assignment


class PairRecord:
    def __init__(self, n_value, group_id, pair_id, sat_pred, unsat_pred, sat_assignment_satisfied):
        self.n_value = n_value
        self.group_id = group_id
        self.pair_id = pair_id
        self.sat_pred = sat_pred
        self.unsat_pred = unsat_pred
        self.sat_assignment_satisfied = sat_assignment_satisfied


def normalize_label(raw: Optional[str]) -> Optional[str]:
    if not raw:
        return None
    s = str(raw).strip().upper()
    if s in {SAT, "SAT"}:
        return SAT
    if s in {UNSAT, "UNSAT"}:
        return UNSAT
    return None


def read_json(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    if not isinstance(obj, dict):
        raise ValueError("{} is not a JSON object".format(path))
    return obj


def extract_pred_from_record(rec: Dict) -> Optional[str]:
    parsed = rec.get("parsed_response")
    if isinstance(parsed, dict):
        pred = normalize_label(parsed.get("result"))
        if pred:
            return pred

    raw = rec.get("raw_response")
    if isinstance(raw, str):
        match = re.search(r'"result"\s*:\s*"([^"]+)"', raw, flags=re.IGNORECASE)
        if match:
            pred = normalize_label(match.group(1))
            if pred:
                return pred
        upper = raw.upper()
        if UNSAT in upper:
            return UNSAT
        if SAT in upper:
            return SAT
    return None


def extract_assignment_from_record(rec: Dict) -> Optional[Dict]:
    parsed = rec.get("parsed_response")
    if isinstance(parsed, dict) and isinstance(parsed.get("assignment"), dict):
        return parsed.get("assignment")

    raw = rec.get("raw_response")
    if isinstance(raw, str):
        try:
            obj = json.loads(raw)
        except Exception:
            return None
        if isinstance(obj, dict) and isinstance(obj.get("assignment"), dict):
            return obj.get("assignment")
    return None


def infer_true_label_from_filename(file_name: str) -> Optional[str]:
    s = file_name.lower()
    if "_unsat" in s:
        return UNSAT
    if "_sat_" in s or "_sat." in s or "_sat_" in s.replace("sat_minfix", "sat"):
        return SAT
    if "_sat_minfix" in s:
        return SAT
    return None


def infer_pair_id_from_filename(file_name: str) -> Optional[str]:
    match = re.search(r"_v(\d+)\.cnf(?:\.json)?$", file_name, flags=re.IGNORECASE)
    if match:
        return "v{:02d}".format(int(match.group(1)))
    if file_name.lower().endswith(".cnf") or file_name.lower().endswith(".cnf.json"):
        return "original"
    return None


def iter_json_files(root: str) -> Iterable[str]:
    for walk_root, _, files in os.walk(root):
        for name in files:
            if name.lower().endswith(".cnf.json"):
                yield os.path.join(walk_root, name)


def print_progress(done: int, total: int, accepted: int, dropped: int, width: int = 30) -> None:
    if total <= 0:
        line = "[------------------------------] 0.0% (0/0) accepted=0 dropped=0"
    else:
        ratio = done / total
        filled = int(width * ratio)
        bar = "#" * filled + "-" * (width - filled)
        line = "[{}] {:5.1f}% ({}/{}) accepted={} dropped={}".format(
            bar, ratio * 100, done, total, accepted, dropped
        )
    print("\r" + line, end="", flush=True)


def parse_instance(input_root: str, json_path: str) -> Tuple[Optional[InstanceRecord], Optional[str]]:
    try:
        rec = read_json(json_path)
    except Exception:
        return None, "json_read_error"

    source_file = rec.get("source_file") if isinstance(rec.get("source_file"), str) else None
    parse_path = source_file or json_path
    normalized_path = parse_path.replace("\\", "/")
    match = re.search(r"/N_(\d+)/(?:cases|cases_prompt)/([^/]+)/([^/]+?)(?:\.json)?$", normalized_path)
    if not match:
        rel = os.path.relpath(json_path, input_root).replace("\\", "/")
        match = re.search(r"(?:^|/)N_(\d+)/(?:cases|cases_prompt)/([^/]+)/([^/]+)$", rel)
    if not match:
        return None, "path_too_short"

    n_value = int(match.group(1))
    group_id = rec.get("group_id") if isinstance(rec.get("group_id"), str) else match.group(2)
    file_name = match.group(3)
    pair_id = infer_pair_id_from_filename(file_name)
    if not pair_id:
        return None, "invalid_pair_id"
    true_label = infer_true_label_from_filename(file_name)
    if true_label is None:
        return None, "invalid_true_label"

    pred = extract_pred_from_record(rec)
    assignment = extract_assignment_from_record(rec)
    return InstanceRecord(n_value, group_id, pair_id, true_label, pred, json_path, source_file, assignment), None


def canonical_instance_dedup_key(item: InstanceRecord):
    path = item.source_file or item.json_path
    normalized = path.replace("\\", "/")
    marker = "/" + MIN_EDIT_DATASET_DIR + "/"
    if marker in normalized:
        rel = normalized.split(marker, 1)[1]
        if rel.endswith(".json"):
            rel = rel[:-5]
        return ("dataset_path", rel)

    file_name = os.path.basename(normalized)
    if file_name.endswith(".json"):
        file_name = file_name[:-5]
    return ("n_file", item.n_value, file_name)


def load_instances(input_root: str, show_progress: bool = False) -> Tuple[List[InstanceRecord], Dict[str, int]]:
    stats = {
        "total_json_files_seen": 0,
        "accepted_instances": 0,
        "dropped_path_too_short": 0,
        "dropped_invalid_n_dir": 0,
        "dropped_missing_cases_dir": 0,
        "dropped_invalid_pair_id": 0,
        "dropped_invalid_true_label": 0,
        "dropped_json_read_error": 0,
        "dropped_duplicate_source_file": 0,
        "missing_prediction_label": 0,
    }
    files = list(iter_json_files(input_root))
    out: List[InstanceRecord] = []
    if show_progress:
        print_progress(0, len(files), 0, 0)
    for i, path in enumerate(files, start=1):
        stats["total_json_files_seen"] += 1
        item, err = parse_instance(input_root, path)
        if err:
            stats["dropped_" + err] += 1
        else:
            if item.pred_label is None:
                stats["missing_prediction_label"] += 1
            out.append(item)
            stats["accepted_instances"] += 1
        if show_progress:
            dropped = stats["total_json_files_seen"] - stats["accepted_instances"]
            print_progress(i, len(files), stats["accepted_instances"], dropped)
    if show_progress and files:
        print()
    deduped: List[InstanceRecord] = []
    seen = set()
    for item in out:
        key = canonical_instance_dedup_key(item)
        if key in seen:
            stats["dropped_duplicate_source_file"] += 1
            continue
        seen.add(key)
        deduped.append(item)
    stats["accepted_instances"] = len(deduped)
    return deduped, stats


def build_pairs(instances: List[InstanceRecord]) -> Tuple[List[PairRecord], Dict[str, int]]:
    bucket = defaultdict(lambda: {"sat_pred": None, "unsat_pred": None, "sat_assignment_satisfied": False})
    stats = {"pairs_total": 0, "duplicate_conflict_sat": 0, "duplicate_conflict_unsat": 0}

    for item in instances:
        key = (item.n_value, item.group_id, item.pair_id)
        slot = "sat_pred" if item.true_label == SAT else "unsat_pred"
        old = bucket[key][slot]
        if old is not None and item.pred_label is not None and old != item.pred_label:
            stats["duplicate_conflict_sat" if slot == "sat_pred" else "duplicate_conflict_unsat"] += 1
        if item.pred_label is not None:
            bucket[key][slot] = item.pred_label
            if item.true_label == SAT and item.pred_label == SAT:
                bucket[key]["sat_assignment_satisfied"] = verify_assignment_for_instance(item) == "satisfied"

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
    if den == 0:
        return None
    return num / den


def _binary_stats(instances: List[InstanceRecord], positive_label: str) -> Dict[str, int]:
    tp = fp = fn = tn = 0
    for item in instances:
        if item.pred_label is None:
            continue
        pred_pos = item.pred_label == positive_label
        true_pos = item.true_label == positive_label
        if pred_pos and true_pos:
            tp += 1
        elif pred_pos and not true_pos:
            fp += 1
        elif (not pred_pos) and true_pos:
            fn += 1
        else:
            tn += 1
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn}


def resolve_source_file(path: Optional[str]) -> Optional[str]:
    if not path:
        return None
    candidates = [path]
    if path.startswith("/work/"):
        candidates.append("/ddnB" + path)
    marker = "SAT_Group_Evaluation/"
    if marker in path:
        rel = path.split(marker, 1)[1]
        candidates.append(os.path.join(REPO_ROOT, rel))
    for candidate in candidates:
        if candidate and os.path.isfile(candidate):
            return candidate
    return None


def parse_cnf_file(path: str) -> Tuple[List[List[int]], int]:
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()

    header_vars = 0
    header_match = re.search(r"(?m)^\s*p\s+cnf\s+(\d+)\s+\d+\s*$", text)
    if header_match:
        header_vars = int(header_match.group(1))

    symbolic_clauses: List[List[int]] = []
    for raw_clause in re.findall(r"\(([^()]*)\)", text):
        if "∨" not in raw_clause and " v " not in raw_clause.lower():
            continue
        literals = re.findall(r"[-−]?\d+", raw_clause)
        if not literals:
            continue
        clause = [int(lit.replace("−", "-")) for lit in literals]
        symbolic_clauses.append(clause)

    if symbolic_clauses:
        max_var = max((abs(lit) for clause in symbolic_clauses for lit in clause), default=0)
        return symbolic_clauses, max(header_vars, max_var)

    clauses: List[List[int]] = []
    num_vars = header_vars
    current: List[int] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("c") or line.startswith("p "):
            continue
        tokens = line.split()
        if not tokens or any(not re.fullmatch(r"[-+]?\d+", token) for token in tokens):
            continue
        for token in tokens:
            lit = int(token)
            if lit == 0:
                if current:
                    clauses.append(current)
                    current = []
            else:
                current.append(lit)
                num_vars = max(num_vars, abs(lit))
    if current:
        clauses.append(current)
    return clauses, num_vars


def normalize_assignment(raw_assignment: Optional[Dict]) -> Optional[Dict[int, bool]]:
    if not isinstance(raw_assignment, dict):
        return None
    out: Dict[int, bool] = {}
    for raw_key, raw_value in raw_assignment.items():
        try:
            var = abs(int(str(raw_key).strip()))
        except Exception:
            return None
        if var <= 0:
            return None

        if isinstance(raw_value, bool):
            value = raw_value
        elif isinstance(raw_value, str):
            lowered = raw_value.strip().lower()
            if lowered == "true":
                value = True
            elif lowered == "false":
                value = False
            else:
                return None
        elif raw_value in {0, 1}:
            value = bool(raw_value)
        else:
            return None

        out[var] = value
    return out


def assignment_satisfies_with_cdcl(clauses: List[List[int]], assignment: Dict[int, bool]) -> Optional[bool]:
    """Return whether the original CNF plus the predicted assignment is SAT."""
    assignment_units = [[var if value else -var] for var, value in sorted(assignment.items())]
    assumptions = [unit[0] for unit in assignment_units]

    try:
        from pysat.solvers import Minisat22
    except Exception:
        Minisat22 = None

    if Minisat22 is not None:
        try:
            with Minisat22(bootstrap_with=clauses) as solver:
                return bool(solver.solve(assumptions=assumptions))
        except Exception:
            pass

    try:
        import pycosat
    except Exception:
        return None

    try:
        result = pycosat.solve(clauses + assignment_units)
    except Exception:
        return None
    if result == "UNSAT":
        return False
    if result == "UNKNOWN":
        return None
    return True


def verify_assignment_for_instance(item: InstanceRecord) -> str:
    source_path = resolve_source_file(item.source_file)
    if source_path is None:
        return "missing_cnf"

    assignment = normalize_assignment(item.assignment)
    if assignment is None:
        return "missing_assignment" if item.assignment is None else "invalid_assignment"

    try:
        clauses, num_vars = parse_cnf_file(source_path)
    except Exception:
        return "missing_cnf"

    required_vars = set(range(1, num_vars + 1))
    used_vars = {abs(lit) for clause in clauses for lit in clause}
    required_vars.update(used_vars)
    if not required_vars.issubset(set(assignment.keys())):
        return "invalid_assignment"

    cdcl_result = assignment_satisfies_with_cdcl(clauses, assignment)
    if cdcl_result is None:
        return "cdcl_unavailable"
    return "satisfied" if cdcl_result else "unsatisfied"


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
        if precision_unsat is not None and recall_unsat is not None and (precision_unsat + recall_unsat) > 0
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
        "direct_separability_ability": (
            recall_sat * recall_unsat
            if recall_sat is not None and recall_unsat is not None
            else None
        ),
        "g_mean": (
            ((recall_sat * recall_unsat) ** 0.5)
            if recall_sat is not None and recall_unsat is not None and recall_sat * recall_unsat >= 0
            else None
        ),
        "balanced_accuracy": (
            0.5 * (recall_sat + recall_unsat)
            if recall_sat is not None and recall_unsat is not None
            else None
        ),
        "youden_j": (
            recall_sat + recall_unsat - 1.0
            if recall_sat is not None and recall_unsat is not None
            else None
        ),
        "sat_positive_precision": precision_sat,
        "sat_positive_recall": recall_sat,
        "sat_positive_f1": f1_sat,
        "unsat_positive_precision": precision_unsat,
        "unsat_positive_recall": recall_unsat,
        "unsat_positive_f1": f1_unsat,
    }
    metrics.update(compute_assignment_metrics(instances))
    return metrics


def compute_group_adr_metrics(pairs: List[PairRecord], sgadr_lambda: float = 1.0) -> Dict:
    grouped = defaultdict(list)
    for pair in pairs:
        grouped[pair.group_id].append(pair)

    group_rows = []
    for gid in sorted(grouped):
        plist = grouped[gid]
        k_g = len(plist)
        complete = 0
        success = 0
        success_with_assignment = 0
        for pair in plist:
            if pair.sat_pred is not None and pair.unsat_pred is not None:
                complete += 1
                a = 1 if pair.sat_pred == SAT else 0
                b = 1 if pair.unsat_pred == UNSAT else 0
                pair_success = a * b
                success += pair_success
                success_with_assignment += pair_success * (1 if pair.sat_assignment_satisfied else 0)
        group_adr = _safe_div(success, complete) if complete > 0 else None
        group_adr_with_assignment = _safe_div(success_with_assignment, complete) if complete > 0 else None
        group_variance = group_adr * (1.0 - group_adr) if group_adr is not None else None
        group_variance_with_assignment = (
            group_adr_with_assignment * (1.0 - group_adr_with_assignment)
            if group_adr_with_assignment is not None
            else None
        )
        group_sgadr = (
            group_adr - sgadr_lambda * group_variance
            if group_adr is not None and group_variance is not None
            else None
        )
        group_sgadr_with_assignment = (
            group_adr_with_assignment - sgadr_lambda * group_variance_with_assignment
            if group_adr_with_assignment is not None and group_variance_with_assignment is not None
            else None
        )
        group_rows.append(
            {
                "group_id": gid,
                "k_g_total_pairs": k_g,
                "k_g_complete_pairs": complete,
                "pair_success_count": success,
                "pair_success_with_assignment_count": success_with_assignment,
                "ADR_g": group_adr,
                "ADR_with_assignment_g": group_adr_with_assignment,
                "ADR_variance_g": group_variance,
                "ADR_variance_with_assignment_g": group_variance_with_assignment,
                "SGADR_g": group_sgadr,
                "SGADR_with_assignment_g": group_sgadr_with_assignment,
            }
        )

    total_pairs = sum(x["k_g_total_pairs"] for x in group_rows)
    total_complete = sum(x["k_g_complete_pairs"] for x in group_rows)
    total_success = sum(x["pair_success_count"] for x in group_rows)
    total_success_with_assignment = sum(x["pair_success_with_assignment_count"] for x in group_rows)
    valid_group_rows = [x for x in group_rows if x["ADR_g"] is not None]
    valid_group_adrs = [x["ADR_g"] for x in valid_group_rows]
    valid_group_adrs_with_assignment = [x["ADR_with_assignment_g"] for x in valid_group_rows]
    valid_group_sgadrs = [x["SGADR_g"] for x in valid_group_rows if x["SGADR_g"] is not None]
    valid_group_sgadrs_with_assignment = [
        x["SGADR_with_assignment_g"] for x in valid_group_rows if x["SGADR_with_assignment_g"] is not None
    ]
    gadr = _safe_div(sum(valid_group_adrs), len(valid_group_adrs)) if valid_group_adrs else None
    gadr_with_assignment = (
        _safe_div(sum(valid_group_adrs_with_assignment), len(valid_group_adrs_with_assignment))
        if valid_group_adrs_with_assignment
        else None
    )
    sgadr = _safe_div(sum(valid_group_sgadrs), len(valid_group_sgadrs)) if valid_group_sgadrs else None
    sgadr_with_assignment = (
        _safe_div(sum(valid_group_sgadrs_with_assignment), len(valid_group_sgadrs_with_assignment))
        if valid_group_sgadrs_with_assignment
        else None
    )
    adr = _safe_div(total_success, total_complete) if total_complete > 0 else None
    adr_with_assignment = _safe_div(total_success_with_assignment, total_complete) if total_complete > 0 else None
    sgadr_star = adr - sgadr_lambda * adr * (1.0 - adr) if adr is not None else None
    sgadr_star_with_assignment = (
        adr_with_assignment - sgadr_lambda * adr_with_assignment * (1.0 - adr_with_assignment)
        if adr_with_assignment is not None
        else None
    )
    threshold_group_adrs = {
        "group_level_ADR_at_least_{}_pair_success".format(threshold): (
            _safe_div(
                sum(1 for row in valid_group_rows if row["pair_success_count"] >= threshold),
                len(valid_group_rows),
            )
            if valid_group_rows
            else None
        )
        for threshold in PAIR_SUCCESS_THRESHOLDS
    }

    metrics = {
        "sgadr_lambda": sgadr_lambda,
        "num_groups_total": len(group_rows),
        "num_groups_with_complete_pairs": len(valid_group_adrs),
        "num_pairs_total": total_pairs,
        "num_complete_pairs_total": total_complete,
        "pair_success_both_correct_total": total_success,
        "pair_success_with_assignment_total": total_success_with_assignment,
        "GADR": gadr,
        "ADR": adr,
        "SGADR": sgadr,
        "SGADR_star": sgadr_star,
        "GADR_with_assignment": gadr_with_assignment,
        "ADR_with_assignment": adr_with_assignment,
        "SGADR_with_assignment": sgadr_with_assignment,
        "SGADR_star_with_assignment": sgadr_star_with_assignment,
        "groups": group_rows,
    }
    metrics.update(threshold_group_adrs)
    metrics.update(compute_strict_cluster_metrics(pairs, adr))
    return metrics


def compute_strict_cluster_metrics(pairs: List[PairRecord], adr: Optional[float]) -> Dict:
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


def compute_by_n(pairs: List[PairRecord], instances: List[InstanceRecord], sgadr_lambda: float = 1.0) -> List[Dict]:
    pairs_by_n = defaultdict(list)
    inst_by_n = defaultdict(list)
    for pair in pairs:
        pairs_by_n[pair.n_value].append(pair)
    for item in instances:
        inst_by_n[item.n_value].append(item)

    rows = []
    for n_value in sorted(set(list(pairs_by_n.keys()) + list(inst_by_n.keys()))):
        pair_metrics = compute_group_adr_metrics(pairs_by_n.get(n_value, []), sgadr_lambda=sgadr_lambda)
        inst_metrics = compute_instance_metrics(inst_by_n.get(n_value, []))
        rows.append({"N": n_value, "metrics": dict(pair_metrics, **inst_metrics)})
    return rows


def write_csv(path: str, headers: List[str], rows: List[Dict]) -> None:
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_excel(path: str, by_n_rows: List[Dict], overall_rows: List[Dict]) -> Tuple[bool, Optional[str]]:
    try:
        import pandas as pd
    except Exception as exc:
        return False, "pandas not available ({})".format(exc)
    try:
        by_n_df = pd.DataFrame(sorted(by_n_rows, key=lambda x: (str(x["model"]), int(x["N"]))))
        overall_df = pd.DataFrame(sorted(overall_rows, key=lambda x: str(x["model"])))
        with pd.ExcelWriter(path) as writer:
            by_n_df.to_excel(writer, sheet_name="by_N", index=False)
            overall_df.to_excel(writer, sheet_name="overall", index=False)
        return True, None
    except Exception as exc:
        return False, str(exc)


def build_direct_classification_summary_rows(by_n_rows: List[Dict]) -> List[Dict]:
    return [{key: row.get(key) for key in DIRECT_CLASSIFICATION_SUMMARY_HEADERS} for row in by_n_rows]


def build_direct_classification_summary_overall_rows(overall_rows: List[Dict]) -> List[Dict]:
    return [
        {key: row.get(key) for key in DIRECT_CLASSIFICATION_SUMMARY_OVERALL_HEADERS}
        for row in overall_rows
    ]


def write_direct_classification_summary_excel(
    path: str,
    by_n_rows: List[Dict],
    overall_rows: List[Dict],
) -> Tuple[bool, Optional[str]]:
    try:
        import pandas as pd
    except Exception as exc:
        return False, "pandas not available ({})".format(exc)
    try:
        by_n_df = pd.DataFrame(sorted(by_n_rows, key=lambda x: (str(x["model"]), int(x["N"]))))
        overall_df = pd.DataFrame(sorted(overall_rows, key=lambda x: str(x["model"])))
        with pd.ExcelWriter(path) as writer:
            by_n_df.to_excel(writer, sheet_name="by_N", index=False)
            overall_df.to_excel(writer, sheet_name="overall", index=False)
        return True, None
    except Exception as exc:
        return False, str(exc)


def build_pairwise_preview(instances: List[InstanceRecord], limit: int) -> List[Dict]:
    bucket = defaultdict(lambda: {"sat_instance": None, "unsat_instance": None})
    for item in instances:
        key = (item.n_value, item.group_id, item.pair_id)
        payload = {
            "true_label": item.true_label,
            "prediction_result": item.pred_label,
            "json_path": item.json_path,
            "source_file": item.source_file,
        }
        if item.true_label == SAT:
            bucket[key]["sat_instance"] = payload
        else:
            bucket[key]["unsat_instance"] = payload

    out = []
    for n_value, group_id, pair_id in sorted(bucket.keys()):
        out.append(
            {
                "N": n_value,
                "group_id": group_id,
                "pair_id": pair_id,
                "instances": [
                    bucket[(n_value, group_id, pair_id)]["sat_instance"],
                    bucket[(n_value, group_id, pair_id)]["unsat_instance"],
                ],
            }
        )
        if len(out) >= limit:
            break
    return out


def _sanitize_model_label(raw: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "__", raw).strip("_")


def _is_n_dir(part: str) -> bool:
    return re.fullmatch(r"N_\d+", part) is not None


def peek_model_name(dataset_root: str) -> Optional[str]:
    for path in iter_json_files(dataset_root):
        try:
            rec = read_json(path)
        except Exception:
            continue
        model = rec.get("model")
        if isinstance(model, str) and model:
            return model
    return None


def infer_claude_opus_4_7_variant(after_outputs: List[str], run_dir: str) -> Optional[str]:
    if not after_outputs or after_outputs[0] != "claude-opus-4-7":
        return None

    tail = after_outputs[1:]
    if tail and _is_n_dir(tail[0]):
        tail = tail[1:]
    if tail:
        return tail[0]

    run_base = os.path.basename(run_dir)
    if run_base and run_base != "claude-opus-4-7" and not _is_n_dir(run_base):
        return run_base
    return None


def infer_model_label(commercial_root: str, run_dir: str, dataset_root: str) -> str:
    rel = os.path.relpath(run_dir, commercial_root).replace("\\", "/")
    parts = rel.split("/")
    provider = parts[0] if parts else "commercial_api"
    model = None
    variant = None
    if "outputs" in parts:
        after_outputs = parts[parts.index("outputs") + 1 :]
        if after_outputs and "batch_groups" not in after_outputs[0]:
            model = after_outputs[0]
            if provider == "claude":
                variant = infer_claude_opus_4_7_variant(after_outputs, run_dir)
    if not model:
        model = peek_model_name(dataset_root) or os.path.basename(run_dir)
    if variant:
        model = "{}__{}".format(model, variant)
    return _sanitize_model_label("{}__{}".format(provider, model))


def find_commercial_input_groups(commercial_root: str) -> List[Tuple[str, List[str]]]:
    grouped = defaultdict(list)
    if not os.path.isdir(commercial_root):
        return []

    for walk_root, dirs, _ in os.walk(commercial_root):
        if os.path.basename(walk_root) != "by_file":
            continue
        dataset_root = None
        for root, subdirs, _ in os.walk(walk_root):
            if os.path.basename(root) == MIN_EDIT_DATASET_DIR:
                dataset_root = os.path.abspath(root)
                subdirs[:] = []
                break
        if dataset_root is None:
            dirs[:] = []
            continue
        run_dir = os.path.dirname(walk_root)
        label = infer_model_label(commercial_root, run_dir, dataset_root)
        grouped[label].append(dataset_root)
        dirs[:] = []

    return [(model, sorted(set(roots))) for model, roots in sorted(grouped.items())]


def merge_load_stats(stats_list: List[Dict[str, int]], duplicate_count: int) -> Dict[str, int]:
    out = defaultdict(int)
    for stats in stats_list:
        for key, value in stats.items():
            out[key] += value
    out["dropped_duplicate_source_file"] += duplicate_count
    out["accepted_instances"] = out["accepted_instances"] - duplicate_count
    return dict(out)


def load_instances_from_roots(input_roots: List[str], show_progress: bool = False) -> Tuple[List[InstanceRecord], Dict[str, int]]:
    all_instances: List[InstanceRecord] = []
    stats_list: List[Dict[str, int]] = []
    for input_root in input_roots:
        instances, stats = load_instances(input_root, show_progress=show_progress)
        all_instances.extend(instances)
        stats_list.append(stats)

    deduped: List[InstanceRecord] = []
    seen = set()
    duplicate_count = 0
    for item in all_instances:
        key = canonical_instance_dedup_key(item)
        if key in seen:
            duplicate_count += 1
            continue
        seen.add(key)
        deduped.append(item)

    return deduped, merge_load_stats(stats_list, duplicate_count)


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


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compute zero-shot Group ADR and traditional metrics for commercial API "
            "model outputs with the same metric logic and CSV column order as zero_shot_metrics."
        )
    )
    parser.add_argument("--commercial-api-root", default=DEFAULT_COMMERCIAL_API_ROOT)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--preview-limit", type=int, default=40)
    parser.add_argument("--sgadr-lambda", type=float, default=1.0)
    parser.add_argument(
        "--show-progress",
        action="store_true",
        help="Print per-file progress while loading each model output tree.",
    )
    args = parser.parse_args()

    commercial_api_root = os.path.abspath(args.commercial_api_root)
    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    roots = find_commercial_input_groups(commercial_api_root)
    if not roots:
        raise SystemExit(f"No commercial API min_edit_3sat_dataset roots found under {commercial_api_root}")

    all_reports: List[Dict] = []
    by_n_rows: List[Dict] = []
    overall_rows: List[Dict] = []
    pairwise_preview_by_model: Dict[str, List[Dict]] = {}

    for model, input_roots in roots:
        print(f"[info] loading model={model} roots={len(input_roots)}")
        instances, load_stats = load_instances_from_roots(input_roots, show_progress=args.show_progress)
        pairs, pair_stats = build_pairs(instances)
        overall_pair = compute_group_adr_metrics(pairs, sgadr_lambda=args.sgadr_lambda)
        overall_inst = compute_instance_metrics(instances)
        overall = {**overall_pair, **overall_inst}
        by_n = compute_by_n(pairs, instances, sgadr_lambda=args.sgadr_lambda)
        pairwise_preview = build_pairwise_preview(instances, args.preview_limit)

        all_reports.append(
            {
                "model": model,
                "input_root": ";".join(input_roots),
                "load_stats": load_stats,
                "pair_stats": pair_stats,
                "overall_metrics": overall,
                "metrics_by_N": by_n,
            }
        )
        pairwise_preview_by_model[model] = pairwise_preview

        for row in by_n:
            by_n_rows.append(row_from_metrics(model, row["N"], row["metrics"]))
        overall_rows.append(overall_row_from_metrics(model, ";".join(input_roots), overall))

        print(
            f"[info] done model={model} instances={load_stats['accepted_instances']} "
            f"pairs={pair_stats['pairs_total']} groups={overall['num_groups_total']}"
        )

    report_json = os.path.join(output_dir, "commercial_api_group_adr_report.json")
    with open(report_json, "w", encoding="utf-8") as f:
        json.dump(
            {
                "commercial_api_root": commercial_api_root,
                "input_roots_by_model": {model: input_roots for model, input_roots in roots},
                "sgadr_lambda": args.sgadr_lambda,
                "models": all_reports,
                "pairwise_preview_by_model": pairwise_preview_by_model,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    by_n_csv = os.path.join(output_dir, "commercial_api_group_adr_report.by_N.csv")
    write_csv(by_n_csv, BY_N_HEADERS, sorted(by_n_rows, key=lambda x: (str(x["model"]), int(x["N"]))))

    overall_csv = os.path.join(output_dir, "commercial_api_group_adr_report.overall.csv")
    write_csv(overall_csv, OVERALL_HEADERS, sorted(overall_rows, key=lambda x: str(x["model"])))

    direct_metric_by_n_rows = build_direct_classification_summary_rows(by_n_rows)
    direct_metric_overall_rows = build_direct_classification_summary_overall_rows(overall_rows)
    direct_metric_by_n_csv = os.path.join(
        output_dir, "commercial_api_group_adr_report.Existing_Category_Metrics.by_N.csv"
    )
    write_csv(
        direct_metric_by_n_csv,
        DIRECT_CLASSIFICATION_SUMMARY_HEADERS,
        sorted(direct_metric_by_n_rows, key=lambda x: (str(x["model"]), int(x["N"]))),
    )
    direct_metric_overall_csv = os.path.join(
        output_dir, "commercial_api_group_adr_report.Existing_Category_Metrics.overall.csv"
    )
    write_csv(
        direct_metric_overall_csv,
        DIRECT_CLASSIFICATION_SUMMARY_OVERALL_HEADERS,
        sorted(direct_metric_overall_rows, key=lambda x: str(x["model"])),
    )

    excel_path = os.path.join(output_dir, "commercial_api_group_adr_report.xlsx")
    excel_written, excel_error = write_excel(excel_path, by_n_rows, overall_rows)
    if not excel_written:
        print(f"[warn] excel export skipped: {excel_error}")

    direct_metric_excel_path = os.path.join(
        output_dir, "commercial_api_group_adr_report.Existing_Category_Metrics.xlsx"
    )
    direct_metric_excel_written, direct_metric_excel_error = write_direct_classification_summary_excel(
        direct_metric_excel_path,
        direct_metric_by_n_rows,
        direct_metric_overall_rows,
    )
    if not direct_metric_excel_written:
        print(f"[warn] direct classification metric excel export skipped: {direct_metric_excel_error}")

    preview_json = os.path.join(output_dir, "commercial_api_group_adr_report.pairwise_preview.json")
    with open(preview_json, "w", encoding="utf-8") as f:
        json.dump(pairwise_preview_by_model, f, ensure_ascii=False, indent=2)

    print(
        json.dumps(
            {
                "commercial_api_root": commercial_api_root,
                "num_models": len(roots),
                "models": [model for model, _ in roots],
                "output_dir": output_dir,
                "report_json": report_json,
                "by_n_csv": by_n_csv,
                "overall_csv": overall_csv,
                "direct_metric_by_n_csv": direct_metric_by_n_csv,
                "direct_metric_overall_csv": direct_metric_overall_csv,
                "excel_xlsx": excel_path if excel_written else None,
                "direct_metric_excel_xlsx": direct_metric_excel_path if direct_metric_excel_written else None,
                "pairwise_preview_json": preview_json,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
