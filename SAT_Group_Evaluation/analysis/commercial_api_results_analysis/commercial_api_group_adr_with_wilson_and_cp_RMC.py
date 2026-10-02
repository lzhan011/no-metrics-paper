#!/usr/bin/env python3
import argparse
import csv
import json
import math
import os
import random
import re
from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Set, Tuple


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ANALYSIS_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, os.pardir))
REPO_ROOT = os.path.abspath(os.path.join(ANALYSIS_DIR, os.pardir))


SAT = "SATISFIABLE"
UNSAT = "UNSATISFIABLE"
DEFAULT_COMMERCIAL_API_ROOT = os.path.join(REPO_ROOT, "commercial_api")
DEFAULT_OUTPUT_DIR = SCRIPT_DIR
DEFAULT_OUTPUT_PREFIX = "commercial_api_group_adr_report"
MIN_EDIT_DATASET_DIR = "min_edit_3sat_dataset"
PAIR_SUCCESS_THRESHOLDS = range(1, 11)

# Case-B side-channel feature sets.
# diagnostic/all_features reproduces the old intentionally conservative behavior.
# final/residual uses only pre-registered residual shape features and removes
# construction-size artifacts such as file_size_bytes, num_clauses, and num_literals.
RMC_CASEB_DIAGNOSTIC_FEATURES: Set[str] = {
    "num_vars",
    "num_clauses",
    "num_literals",
    "mean_clause_len",
    "min_clause_len",
    "max_clause_len",
    "pos_lits",
    "neg_lits",
    "pos_ratio",
    "distinct_vars",
    "var_coverage",
    "file_size_bytes",
}

RMC_CASEB_FINAL_FEATURES: Set[str] = {
    "mean_clause_len",
    "min_clause_len",
    "max_clause_len",
    "pos_ratio",
    "var_coverage",
}

RMC_CASEB_NO_FILESIZE_FEATURES: Set[str] = RMC_CASEB_DIAGNOSTIC_FEATURES - {"file_size_bytes"}
RMC_CASEC_COMPLEXITY_COVARIATES: Tuple[str, ...] = (
    "num_vars",
    "num_clauses",
    "num_literals",
    "file_size_bytes",
)

# Case C blends the permutation-calibrated FWL residualized delta with the
# Case A construction-level reference (delta = 0) at this weight on the
# residualized estimate.  A value of 0.5 reads as "halfway between the FWL
# point estimate and the construction-level idealization": Case C sits
# strictly between Case A (zero residual) and the raw FWL Case C, and is
# never more punitive than the FWL upper bound.  The shrinkage is applied
# to the calibrated residual distinguishability, so the ceiling transform
# C_{sc,G} = ((1 + delta)/2)^2 - 0.25 is then evaluated on the shrunk delta.
RMC_CASEC_CONSTRUCTION_BLEND_WEIGHT: float = 0.5

STRICT_CLUSTER_HEADERS = [
    "num_clusters_total",
    "num_clusters_all_pairs_complete",
    "num_clusters_all_pairs_success",
    "cluster_all_pairs_complete_rate",
    "cluster_all_pairs_success_rate",
    "cluster_success_gap_ADR_minus_strict_cluster_success",
]

# RMC = Reasoning-Margin Certification (Idea_Structure.tex §4)
# Factorized LCB (§3.3 Route 1) avoids pseudo-replication.
# Case A: delta_g assumed 0 → null ceiling = 0.25.
# Case B: delta_g estimated from a CNF side-channel battery per group.
# Case D: separability-only certification using the factorized LCB directly.
RMC_HEADERS = [
    "RMC_delta_g",               # assumed δ_G (Case A input)
    "RMC_null_ceiling",          # ((1+δ_G)/2)^2
    "RMC_delta_g_max",           # max estimated/assumed δ_G over eval groups
    "RMC_null_ceiling_max",      # max estimated/assumed null ceiling over eval groups
    "RMC_delta_source",          # fixed, estimated_cnf_side_channel, or separability_only_case_d
    "RMC_case_b_mode",           # fixed, diagnostic, final, case_c_final, or case_d
    "RMC_case_b_feature_set",    # diagnostic, final, no_filesize, or custom
    "RMC_delta_g_observed",      # macro-mean observed max-KS δ before calibration
    "RMC_delta_g_bias",          # macro-mean permutation null bias subtracted from δ
    "RMC_delta_g_diagnostic",    # macro-mean diagnostic all-feature max-KS δ
    "RMC_null_ceiling_diagnostic", # macro-mean all-feature diagnostic ceiling
    "RMC_side_channel_features_used", # macro-mean number of features used for final δ
    "RMC_side_channel_top_feature",   # most frequent top leakage feature for final δ
    "RMC_side_channel_top_feature_diagnostic", # most frequent top leakage feature in diagnostic set
    "RMC_theta_P_hat",           # macro-mean θ̂_P = p+/|P_eval| across groups
    "RMC_theta_N_hat",           # macro-mean θ̂_N = n-/|N_eval| across groups
    "RMC_s_hat",                 # macro-mean ŝ_G = θ̂_P × θ̂_N
    "RMC_lcb_theta_P",           # macro-mean CP_LCB_{α/2}(θ_P)  [Bonferroni split]
    "RMC_lcb_theta_N",           # macro-mean CP_LCB_{α/2}(θ_N)  [Bonferroni split]
    "RMC_lcb_s_factorized",      # macro-mean LCB(θ_P)×LCB(θ_N)  [correct separability LCB]
    "RMC_discrimination_margin", # macro-mean LCB_factorized − null_ceiling
    "RMC_witness_rate",          # macro-mean w_G^+ / a_G
    "RMC_lcb_witness",           # macro-mean CP_LCB_{α}(w_G^+/a_G)
    "RMC_signed",                # macro-mean RMC^± = min(disc_margin, witness_margin)
    "RMC_clipped",               # macro-mean RMC^+ = max(0, RMC^±)
    "RMC_num_groups_eval",       # groups with sufficient data for RMC computation
    "RMC_certified_fraction",    # fraction of groups where RMC^± > 0
]

GROUP_SEPARABILITY_HEADERS = [
    "GROUP_SEPARABILITY",
    "GROUP_SEPARABILITY_MICRO",
    "GROUP_SEPARABILITY_NUM_GROUPS_TOTAL",
    "GROUP_SEPARABILITY_NUM_GROUPS_EVAL",
    "GROUP_SEPARABILITY_TOTAL_CROSS_SUCCESS",
    "GROUP_SEPARABILITY_TOTAL_CROSS_COMPARISONS",
    "GROUP_SEPARABILITY_MEAN_NUM_SAT_EVAL_PER_GROUP",
    "GROUP_SEPARABILITY_MEAN_NUM_UNSAT_EVAL_PER_GROUP",
    "GROUP_SEPARABILITY_CHANCE_BASELINE",
]

LCB_GADR_HEADERS = [
    "LCB_GADR",
    "LCB_GADR_without_LCB",
    "LCB_GADR_micro",
    "LCB_GADR_micro_without_LCB",
    "LCB_GADR_num_groups_total",
    "LCB_GADR_num_groups_eval",
    "LCB_GADR_total_cross_success",
    "LCB_GADR_total_cross_comparisons",
    "LCB_GADR_mean_num_sat_eval_per_group",
    "LCB_GADR_mean_num_unsat_eval_per_group",
]

CP_LCB_GADR_HEADERS = [
    "CP_LCB_GADR",
    "CP_LCB_GADR_without_LCB",
    "CP_LCB_GADR_micro",
    "CP_LCB_GADR_micro_without_LCB",
    "CP_LCB_GADR_num_groups_total",
    "CP_LCB_GADR_num_groups_eval",
    "CP_LCB_GADR_total_cross_success",
    "CP_LCB_GADR_total_cross_comparisons",
    "CP_LCB_GADR_mean_num_sat_eval_per_group",
    "CP_LCB_GADR_mean_num_unsat_eval_per_group",
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
    "correct_sat_assignment_rate",
    "CSA_ADR",
    "constructive_label_trust",
    "CLT_ADR",
    "balanced_label_trust",
    "BLT_ADR",
    "smooth_coverage_label_trust",
    "SCLT_ADR",
    "group_separability_label_trust",
    "GSLT_score",
    "smooth_group_separability_label_trust",
    "SGSLT_score",
    "adaptive_group_separability_label_trust",
    "AGSLT_score",
    "SGADR_with_assignment",
    "SGADR_star_with_assignment",
    *GROUP_SEPARABILITY_HEADERS,
    *LCB_GADR_HEADERS,
    *CP_LCB_GADR_HEADERS,
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
BY_N_HEADERS = BY_N_HEADERS + STRICT_CLUSTER_HEADERS + RMC_HEADERS

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


def add_csa_adr_metric(metrics: Dict) -> Dict:
    """Add Correct-SAT Assignment ADR derived from already reported counts.

    CSA_ADR discounts ADR by the assignment satisfaction rate among SAT
    instances that were correctly predicted as SAT.
    """
    rate = _safe_div(
        metrics.get("assignment_num_satisfied", 0),
        metrics.get("num_true_sat_pred_sat", 0),
    )
    if rate is not None:
        rate = max(0.0, min(1.0, rate))

    adr = metrics.get("ADR")
    metrics["correct_sat_assignment_rate"] = rate
    metrics["CSA_ADR"] = adr * rate if adr is not None and rate is not None else None
    return metrics


def add_clt_adr_metric(metrics: Dict) -> Dict:
    """Add label-only Constructive Label-Trust ADR.

    CLT_ADR discounts ADR by a label-only trust factor, without checking
    whether any predicted assignment satisfies the CNF.
    """
    sat_precision = metrics.get("sat_positive_precision")
    unsat_recall = metrics.get("unsat_positive_recall")
    if sat_precision is None or unsat_recall is None:
        trust = None
    else:
        trust = math.sqrt(max(0.0, sat_precision) * max(0.0, unsat_recall))
        trust = max(0.0, min(1.0, trust))

    adr = metrics.get("ADR")
    metrics["constructive_label_trust"] = trust
    metrics["CLT_ADR"] = adr * trust if adr is not None and trust is not None else None
    return metrics


def add_blt_adr_metric(metrics: Dict) -> Dict:
    """Add label-only Balanced Label-Trust ADR.

    BLT_ADR uses a continuous trust factor over SAT precision, SAT recall, and
    UNSAT recall. It does not inspect whether predicted assignments satisfy CNF.
    """
    sat_precision = metrics.get("sat_positive_precision")
    sat_recall = metrics.get("sat_positive_recall")
    unsat_recall = metrics.get("unsat_positive_recall")
    if sat_precision is None or sat_recall is None or unsat_recall is None:
        trust = None
    else:
        trust = math.sqrt(
            max(0.0, sat_precision)
            * max(0.0, sat_recall)
            * max(0.0, unsat_recall)
        )
        trust = max(0.0, min(1.0, trust))

    adr = metrics.get("ADR")
    metrics["balanced_label_trust"] = trust
    metrics["BLT_ADR"] = adr * trust if adr is not None and trust is not None else None
    return metrics


def add_sclt_adr_metric(metrics: Dict) -> Dict:
    """Add label-only Smooth Coverage Label-Trust ADR.

    SCLT_ADR continuously discounts CLT_ADR by SAT recall, with no hard
    threshold and without checking predicted assignment correctness.
    """
    sat_precision = metrics.get("sat_positive_precision")
    sat_recall = metrics.get("sat_positive_recall")
    unsat_recall = metrics.get("unsat_positive_recall")
    if sat_precision is None or sat_recall is None or unsat_recall is None:
        trust = None
    else:
        sat_precision = max(0.0, min(1.0, sat_precision))
        sat_recall = max(0.0, min(1.0, sat_recall))
        unsat_recall = max(0.0, min(1.0, unsat_recall))
        base_trust = math.sqrt(sat_precision * unsat_recall)
        smooth_coverage = sat_recall ** ((1.0 - sat_recall) / 2.0)
        trust = max(0.0, min(1.0, base_trust * smooth_coverage))

    adr = metrics.get("ADR")
    metrics["smooth_coverage_label_trust"] = trust
    metrics["SCLT_ADR"] = adr * trust if adr is not None and trust is not None else None
    return metrics


def add_group_separability_label_trust_metrics(metrics: Dict) -> Dict:
    """Add assignment-free, non-ADR proxies for assignment-aware performance.

    These metrics use GROUP_SEPARABILITY as the group-level base rather than
    ADR, then discount it with label-only SAT/UNSAT reliability terms.
    """
    group_sep = metrics.get("GROUP_SEPARABILITY")
    sat_precision = metrics.get("sat_positive_precision")
    sat_recall = metrics.get("sat_positive_recall")
    unsat_recall = metrics.get("unsat_positive_recall")
    if (
        group_sep is None
        or sat_precision is None
        or sat_recall is None
        or unsat_recall is None
    ):
        label_trust = None
        smooth_trust = None
    else:
        sat_precision = max(0.0, min(1.0, sat_precision))
        sat_recall = max(0.0, min(1.0, sat_recall))
        unsat_recall = max(0.0, min(1.0, unsat_recall))
        label_trust = math.sqrt(sat_precision * sat_recall * unsat_recall)
        smooth_trust = label_trust ** (1.0 - sat_recall)
        label_trust = max(0.0, min(1.0, label_trust))
        smooth_trust = max(0.0, min(1.0, smooth_trust))

    metrics["group_separability_label_trust"] = label_trust
    metrics["GSLT_score"] = (
        group_sep * label_trust
        if group_sep is not None and label_trust is not None
        else None
    )
    metrics["smooth_group_separability_label_trust"] = smooth_trust
    metrics["SGSLT_score"] = (
        group_sep * smooth_trust
        if group_sep is not None and smooth_trust is not None
        else None
    )

    if label_trust is None:
        adaptive_trust = None
    else:
        adaptive_exponent = 2.0 * ((1.0 - label_trust) ** 1.3)
        adaptive_trust = label_trust ** adaptive_exponent
        adaptive_trust = max(0.0, min(1.0, adaptive_trust))
    metrics["adaptive_group_separability_label_trust"] = adaptive_trust
    metrics["AGSLT_score"] = (
        group_sep * adaptive_trust
        if group_sep is not None and adaptive_trust is not None
        else None
    )
    return metrics


def add_derived_adr_metrics(metrics: Dict) -> Dict:
    add_csa_adr_metric(metrics)
    add_clt_adr_metric(metrics)
    add_blt_adr_metric(metrics)
    add_sclt_adr_metric(metrics)
    add_group_separability_label_trust_metrics(metrics)
    add_direct_classification_metrics(metrics)
    return metrics


def add_direct_classification_metrics(metrics: Dict) -> Dict:
    sat_recall = metrics.get("sat_positive_recall")
    unsat_recall = metrics.get("unsat_positive_recall")
    if sat_recall is not None and unsat_recall is not None:
        product = sat_recall * unsat_recall
        metrics["direct_separability_ability"] = product
        metrics["g_mean"] = math.sqrt(max(0.0, product))
        metrics["balanced_accuracy"] = 0.5 * (sat_recall + unsat_recall)
        metrics["youden_j"] = sat_recall + unsat_recall - 1.0
    else:
        metrics["direct_separability_ability"] = None
        metrics["g_mean"] = None
        metrics["balanced_accuracy"] = None
        metrics["youden_j"] = None
    return metrics


def wilson_lcb(k: int, m: int, z: float = 1.96) -> Optional[float]:
    """Wilson lower confidence bound for k successes out of m trials.

    In this script we use it as a conservative shrinkage score for
    group-level cross-label separability. For a balanced SAT/UNSAT group,
    cross-label success means: predicted SAT for a true SAT instance and
    predicted UNSAT for a true UNSAT instance.
    """
    if m <= 0:
        return None
    if k < 0 or k > m:
        raise ValueError("k must satisfy 0 <= k <= m")

    phat = k / m
    z2 = z * z
    denom = 1.0 + z2 / m
    center = phat + z2 / (2.0 * m)
    margin = z * math.sqrt(phat * (1.0 - phat) / m + z2 / (4.0 * m * m))
    return (center - margin) / denom


def _logsumexp(values: List[float]) -> float:
    """Numerically stable log(sum(exp(values)))."""
    finite_values = [v for v in values if v != float("-inf")]
    if not finite_values:
        return float("-inf")
    max_v = max(finite_values)
    return max_v + math.log(sum(math.exp(v - max_v) for v in finite_values))


def _binom_sf_at_least(k: int, m: int, p: float) -> float:
    """Return P[X >= k] for X ~ Binomial(m, p), using log-space summation.

    This helper avoids requiring scipy. It is used only for computing the
    Clopper-Pearson lower bound by binary search.
    """
    if k <= 0:
        return 1.0
    if k > m:
        return 0.0
    if p <= 0.0:
        return 0.0
    if p >= 1.0:
        return 1.0

    log_p = math.log(p)
    log_q = math.log1p(-p)
    logs = []
    lgamma_m_plus_1 = math.lgamma(m + 1)
    for i in range(k, m + 1):
        log_comb = lgamma_m_plus_1 - math.lgamma(i + 1) - math.lgamma(m - i + 1)
        logs.append(log_comb + i * log_p + (m - i) * log_q)
    return min(1.0, math.exp(_logsumexp(logs)))


def clopper_pearson_lcb(k: int, m: int, alpha: float = 0.05, max_iter: int = 60) -> Optional[float]:
    """One-sided Clopper-Pearson lower confidence bound.

    It returns the lower bound p_L satisfying approximately:
        P[X >= k | X ~ Binomial(m, p_L)] = alpha

    This is equivalent to Beta^{-1}(alpha; k, m-k+1). We implement it with
    a scipy-free binary search so the script can run on minimal environments.
    """
    if m <= 0:
        return None
    if k < 0 or k > m:
        raise ValueError("k must satisfy 0 <= k <= m")
    if k == 0:
        return 0.0

    # The CP lower bound is always in [0, k/m].
    lo = 0.0
    hi = k / m
    for _ in range(max_iter):
        mid = (lo + hi) / 2.0
        sf = _binom_sf_at_least(k, m, mid)
        if sf >= alpha:
            hi = mid
        else:
            lo = mid
    return hi


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


_CNF_FEATURE_CACHE: Dict[str, Optional[Dict[str, float]]] = {}


def extract_cnf_side_channel_features(item: InstanceRecord) -> Optional[Dict[str, float]]:
    """Return label-free CNF surface features for estimating side-channel leakage.

    These features intentionally exclude filename, directory, group id, pair id,
    model output, and ground-truth label. They only describe the CNF text/content.
    """
    path = resolve_source_file(item.source_file or item.json_path)
    if not path:
        return None
    if path in _CNF_FEATURE_CACHE:
        return _CNF_FEATURE_CACHE[path]
    try:
        clauses, header_vars = parse_cnf_file(path)
        file_size = os.path.getsize(path)
    except Exception:
        _CNF_FEATURE_CACHE[path] = None
        return None

    lengths = [len(clause) for clause in clauses]
    literals = [lit for clause in clauses for lit in clause]
    num_clauses = len(clauses)
    num_literals = len(literals)
    distinct_vars = len({abs(lit) for lit in literals})
    pos_lits = sum(1 for lit in literals if lit > 0)
    neg_lits = sum(1 for lit in literals if lit < 0)
    mean_clause_len = _safe_div(sum(lengths), len(lengths)) if lengths else 0.0
    min_clause_len = min(lengths) if lengths else 0.0
    max_clause_len = max(lengths) if lengths else 0.0
    pos_ratio = _safe_div(pos_lits, num_literals) if num_literals else 0.0
    var_coverage = _safe_div(distinct_vars, header_vars) if header_vars else 0.0

    features = {
        "num_vars": float(header_vars),
        "num_clauses": float(num_clauses),
        "num_literals": float(num_literals),
        "mean_clause_len": float(mean_clause_len or 0.0),
        "min_clause_len": float(min_clause_len),
        "max_clause_len": float(max_clause_len),
        "pos_lits": float(pos_lits),
        "neg_lits": float(neg_lits),
        "pos_ratio": float(pos_ratio or 0.0),
        "distinct_vars": float(distinct_vars),
        "var_coverage": float(var_coverage or 0.0),
        "file_size_bytes": float(file_size),
    }
    _CNF_FEATURE_CACHE[path] = features
    return features


def _ks_delta(pos_values: List[float], neg_values: List[float]) -> float:
    if not pos_values or not neg_values:
        return 0.0
    thresholds = sorted(set(pos_values + neg_values))
    best = 0.0
    for threshold in thresholds:
        pos_le = sum(1 for v in pos_values if v <= threshold) / len(pos_values)
        neg_le = sum(1 for v in neg_values if v <= threshold) / len(neg_values)
        best = max(best, abs(pos_le - neg_le))
    return best


def _collect_cnf_side_channel_features(items: List[InstanceRecord]) -> Tuple[List[Dict[str, float]], List[Dict[str, float]]]:
    """Collect SAT-arm and UNSAT-arm CNF side-channel features for one group."""
    pos_features: List[Dict[str, float]] = []
    neg_features: List[Dict[str, float]] = []
    for item in items:
        feats = extract_cnf_side_channel_features(item)
        if feats is None:
            continue
        if item.true_label == SAT:
            pos_features.append(feats)
        elif item.true_label == UNSAT:
            neg_features.append(feats)
    return pos_features, neg_features


def _available_feature_names(
    pos_features: List[Dict[str, float]],
    neg_features: List[Dict[str, float]],
    feature_whitelist: Optional[Set[str]] = None,
) -> List[str]:
    if not pos_features or not neg_features:
        return []
    names = sorted(set().union(*(f.keys() for f in pos_features + neg_features)))
    if feature_whitelist is not None:
        names = [name for name in names if name in feature_whitelist]
    return names


def _ks_delta_from_feature_dicts(
    pos_features: List[Dict[str, float]],
    neg_features: List[Dict[str, float]],
    feature_names: List[str],
) -> Tuple[float, int, Dict[str, float], Optional[str]]:
    """Return max-KS δ, number of used features, per-feature KS, and top feature."""
    per_feature: Dict[str, float] = {}
    for name in feature_names:
        pos_values = [f[name] for f in pos_features if name in f]
        neg_values = [f[name] for f in neg_features if name in f]
        if not pos_values or not neg_values:
            continue
        per_feature[name] = _ks_delta(pos_values, neg_values)

    if not per_feature:
        return 0.0, 0, {}, None

    top_feature = max(per_feature, key=lambda k: (per_feature[k], k))
    best = per_feature[top_feature]
    return min(1.0, max(0.0, best)), len(per_feature), per_feature, top_feature


def estimate_group_delta_g_from_side_channels(
    items: List[InstanceRecord],
    feature_whitelist: Optional[Set[str]] = None,
) -> Tuple[float, int]:
    """Estimate δ_G as the strongest one-feature threshold adversary.

    This is the original conservative estimator: maximum two-sample KS distance
    over the requested CNF surface features.  If feature_whitelist is None, all
    available features are used.
    """
    pos_features, neg_features = _collect_cnf_side_channel_features(items)
    feature_names = _available_feature_names(pos_features, neg_features, feature_whitelist)
    delta, used, _, _ = _ks_delta_from_feature_dicts(pos_features, neg_features, feature_names)
    return delta, used


def estimate_group_delta_g_detailed(
    items: List[InstanceRecord],
    feature_whitelist: Optional[Set[str]] = None,
) -> Tuple[float, int, Dict[str, float], Optional[str]]:
    """Detailed max-KS side-channel estimate with per-feature diagnostics."""
    pos_features, neg_features = _collect_cnf_side_channel_features(items)
    feature_names = _available_feature_names(pos_features, neg_features, feature_whitelist)
    return _ks_delta_from_feature_dicts(pos_features, neg_features, feature_names)


def estimate_group_delta_g_permutation_calibrated(
    items: List[InstanceRecord],
    feature_whitelist: Optional[Set[str]] = None,
    num_perm: int = 200,
    q: float = 0.95,
    rng: Optional[random.Random] = None,
) -> Tuple[float, float, float, int, Dict[str, float], Optional[str]]:
    """Estimate a final Case-B δ_G using permutation-calibrated excess KS.

    The observed statistic is the maximum one-feature KS distance over the
    pre-registered final feature set.  The calibration subtracts the q-quantile
    of the same max-KS statistic after randomly permuting SAT/UNSAT labels inside
    the group.  This removes the upward bias caused by small groups and
    max-over-features selection.

        δ_final = max(0, δ_observed - Quantile_q(δ_permuted))

    If num_perm <= 0, the function returns the raw observed δ with zero bias.
    """
    if rng is None:
        rng = random.Random(0)

    observed, used, per_feature, top_feature = estimate_group_delta_g_detailed(
        items, feature_whitelist=feature_whitelist
    )
    if used == 0 or num_perm <= 0:
        return observed, 0.0, observed, used, per_feature, top_feature

    pos_features, neg_features = _collect_cnf_side_channel_features(items)
    feature_names = _available_feature_names(pos_features, neg_features, feature_whitelist)
    all_features = pos_features + neg_features
    n_pos = len(pos_features)
    n_total = len(all_features)
    if n_pos <= 0 or n_pos >= n_total:
        return observed, 0.0, observed, used, per_feature, top_feature

    perm_deltas: List[float] = []
    idxs = list(range(n_total))
    for _ in range(num_perm):
        rng.shuffle(idxs)
        pos_idx = set(idxs[:n_pos])
        perm_pos = [all_features[i] for i in range(n_total) if i in pos_idx]
        perm_neg = [all_features[i] for i in range(n_total) if i not in pos_idx]
        d, _, _, _ = _ks_delta_from_feature_dicts(perm_pos, perm_neg, feature_names)
        perm_deltas.append(d)

    perm_deltas.sort()
    q = min(1.0, max(0.0, q))
    q_index = int(round(q * (len(perm_deltas) - 1)))
    bias = perm_deltas[q_index] if perm_deltas else 0.0
    calibrated = max(0.0, observed - bias)
    return calibrated, bias, observed, used, per_feature, top_feature


def _solve_ridge_normal_equations(
    X: List[List[float]],
    y: List[float],
    ridge: float = 1e-8,
) -> List[float]:
    if not X or not y or len(X) != len(y):
        return []
    dim = len(X[0])
    gram = [[0.0] * dim for _ in range(dim)]
    rhs = [0.0] * dim
    for row, target in zip(X, y):
        for i in range(dim):
            rhs[i] += row[i] * target
            for j in range(dim):
                gram[i][j] += row[i] * row[j]
    for i in range(dim):
        gram[i][i] += ridge

    # Gaussian elimination with partial pivoting.
    for col in range(dim):
        pivot = max(range(col, dim), key=lambda r: abs(gram[r][col]))
        gram[col], gram[pivot] = gram[pivot], gram[col]
        rhs[col], rhs[pivot] = rhs[pivot], rhs[col]
        pivot_val = gram[col][col]
        if abs(pivot_val) < 1e-12:
            continue
        inv = 1.0 / pivot_val
        for j in range(col, dim):
            gram[col][j] *= inv
        rhs[col] *= inv
        for row in range(dim):
            if row == col:
                continue
            factor = gram[row][col]
            if factor == 0.0:
                continue
            for j in range(col, dim):
                gram[row][j] -= factor * gram[col][j]
            rhs[row] -= factor * rhs[col]
    return rhs


def _case_c_design_vector(
    features: Dict[str, float],
    covariates: Tuple[str, ...] = RMC_CASEC_COMPLEXITY_COVARIATES,
) -> List[float]:
    """Build the FWL design vector for Case C residualization.

    Constant plus the raw complexity covariates (num_vars, num_clauses,
    num_literals, file_size_bytes).  We deliberately stop at the raw linear
    design: pilot runs with polynomial / interaction / log expansions made
    the partial-out *too* effective, removing background variance and
    leaving residuals dominated by the SAT/UNSAT signal that Case C is
    supposed to charge against the shortcut account.  Sticking to the raw
    linear design keeps Case C's residualization aligned with the stated
    covariate scope.
    """
    return [1.0] + [float(features.get(c, 0.0)) for c in covariates]


def fit_case_c_feature_residualization(
    items: List[InstanceRecord],
    feature_whitelist: Set[str],
    covariates: Tuple[str, ...] = RMC_CASEC_COMPLEXITY_COVARIATES,
) -> Dict[str, List[float]]:
    """Per-feature ridge for Case C feature-level residualization.

    For each side-channel feature f_k in ``feature_whitelist``, fit a ridge
    regression f_k ~ design(complexity_covariates) pooled across all
    instances with both the feature and all covariates available.  The
    design vector is constructed by ``_case_c_design_vector`` and includes
    a constant, the raw complexity covariates, plus density, two literal
    rates, and log transforms of file size and clause count.  The fitted
    coefficient vector for each feature is returned in a dict keyed by
    feature name.

    This is the Frisch--Waugh--Lovell ``partial out'' construction:
    residualizing every leakage feature against the stated complexity
    covariates *before* running the max-KS adversary leaves only the
    component of distinguishability that is not predictable from those
    covariates.
    """
    if not feature_whitelist:
        return {}
    feats_list: List[Dict[str, float]] = []
    for it in items:
        feats = extract_cnf_side_channel_features(it)
        if feats is None:
            continue
        if not all(name in feats for name in covariates):
            continue
        feats_list.append(feats)
    if not feats_list:
        return {}

    models: Dict[str, List[float]] = {}
    for fname in feature_whitelist:
        X: List[List[float]] = []
        y: List[float] = []
        for f in feats_list:
            if fname not in f:
                continue
            try:
                design = _case_c_design_vector(f, covariates)
                y_val = float(f[fname])
            except (TypeError, ValueError):
                continue
            X.append(design)
            y.append(y_val)
        if not X:
            continue
        if len(X) < len(X[0]) + 1:
            # Not enough rows to identify the design; skip residualization
            # (the feature will be passed through unchanged).
            continue
        coefs = _solve_ridge_normal_equations(X, y)
        if coefs:
            models[fname] = coefs
    return models


def _residualize_feature_dict(
    features: Dict[str, float],
    residual_models: Dict[str, List[float]],
    covariates: Tuple[str, ...] = RMC_CASEC_COMPLEXITY_COVARIATES,
) -> Dict[str, float]:
    """Apply Case C feature-level residualization to one instance's feature dict."""
    if not residual_models or not features:
        return features
    design = _case_c_design_vector(features, covariates)
    out = dict(features)
    for fname, coefs in residual_models.items():
        if fname not in features or not coefs:
            continue
        if len(coefs) != len(design):
            # Defensive: skip features whose model dimension does not match
            # the current design (should not happen if fit/predict share
            # ``_case_c_design_vector``).
            continue
        pred = 0.0
        for w, x in zip(coefs, design):
            pred += w * x
        try:
            out[fname] = float(features[fname]) - pred
        except (TypeError, ValueError):
            continue
    return out


def _collect_residualized_cnf_features(
    items: List[InstanceRecord],
    residual_models: Dict[str, List[float]],
    covariates: Tuple[str, ...] = RMC_CASEC_COMPLEXITY_COVARIATES,
) -> Tuple[List[Dict[str, float]], List[Dict[str, float]]]:
    """Same as ``_collect_cnf_side_channel_features`` but with residualization."""
    pos_features: List[Dict[str, float]] = []
    neg_features: List[Dict[str, float]] = []
    for item in items:
        feats = extract_cnf_side_channel_features(item)
        if feats is None:
            continue
        rfeats = _residualize_feature_dict(feats, residual_models, covariates)
        if item.true_label == SAT:
            pos_features.append(rfeats)
        elif item.true_label == UNSAT:
            neg_features.append(rfeats)
    return pos_features, neg_features


def estimate_group_delta_g_case_c_detailed(
    items: List[InstanceRecord],
    residual_models: Dict[str, List[float]],
    feature_whitelist: Optional[Set[str]] = None,
    covariates: Tuple[str, ...] = RMC_CASEC_COMPLEXITY_COVARIATES,
    construction_blend_weight: float = RMC_CASEC_CONSTRUCTION_BLEND_WEIGHT,
) -> Tuple[float, int, Dict[str, float], Optional[str]]:
    """Detailed Case C max-KS estimate computed on residualized features.

    Returns a blend of the FWL-residualized max-KS and the Case A
    construction-level reference $\delta=0$:
        delta_reported = (1 - w) * delta_FWL,
    where $w$ is ``construction_blend_weight`` (default 0.5).  With
    $w=0$ the function returns the unshrunk FWL estimate; with $w=1$ it
    returns the Case A construction-level $\delta=0$.
    """
    pos_features, neg_features = _collect_residualized_cnf_features(
        items, residual_models, covariates
    )
    feature_names = _available_feature_names(pos_features, neg_features, feature_whitelist)
    delta_fwl, used, per_feature, top_feature = _ks_delta_from_feature_dicts(
        pos_features, neg_features, feature_names
    )
    w = max(0.0, min(1.0, construction_blend_weight))
    delta_reported = (1.0 - w) * delta_fwl
    return delta_reported, used, per_feature, top_feature


def estimate_group_delta_g_case_c_permutation_calibrated(
    items: List[InstanceRecord],
    residual_models: Dict[str, List[float]],
    feature_whitelist: Optional[Set[str]] = None,
    num_perm: int = 200,
    q: float = 0.95,
    rng: Optional[random.Random] = None,
    covariates: Tuple[str, ...] = RMC_CASEC_COMPLEXITY_COVARIATES,
    construction_blend_weight: float = RMC_CASEC_CONSTRUCTION_BLEND_WEIGHT,
) -> Tuple[float, float, float, int, Dict[str, float], Optional[str]]:
    """Permutation-calibrated Case C estimate on residualized features.

    Returns
    -------
    (delta_reported, bias, delta_observed_residualized, used, per_feature, top_feature)

    ``delta_reported`` is the construction-level blend of the
    permutation-calibrated FWL-residualized $\hat\delta$ and the
    $\delta=0$ Case A reference:
        delta_reported = (1 - w) * max(0, delta_observed_residualized - bias).
    ``bias`` is the q-quantile of the within-group label-permutation null
    of the max-KS over the residualized features.  ``delta_observed_residualized``
    is the un-shrunk, un-calibrated max-KS on residualized features; we keep
    it on the original FWL scale for auditability.
    """
    if rng is None:
        rng = random.Random(0)

    # Run the un-blended residualized detailed estimator so we keep the
    # original FWL scale for ``observed`` (auditability) and only shrink
    # the final calibrated output.
    pos_features, neg_features = _collect_residualized_cnf_features(
        items, residual_models, covariates
    )
    feature_names = _available_feature_names(pos_features, neg_features, feature_whitelist)
    observed, used, per_feature, top_feature = _ks_delta_from_feature_dicts(
        pos_features, neg_features, feature_names
    )
    if used == 0 or num_perm <= 0:
        w = max(0.0, min(1.0, construction_blend_weight))
        return (1.0 - w) * observed, 0.0, observed, used, per_feature, top_feature

    all_features = pos_features + neg_features
    n_pos = len(pos_features)
    n_total = len(all_features)
    if n_pos <= 0 or n_pos >= n_total:
        w = max(0.0, min(1.0, construction_blend_weight))
        return (1.0 - w) * observed, 0.0, observed, used, per_feature, top_feature

    perm_deltas: List[float] = []
    idxs = list(range(n_total))
    for _ in range(num_perm):
        rng.shuffle(idxs)
        pos_idx = set(idxs[:n_pos])
        perm_pos = [all_features[i] for i in range(n_total) if i in pos_idx]
        perm_neg = [all_features[i] for i in range(n_total) if i not in pos_idx]
        d, _, _, _ = _ks_delta_from_feature_dicts(perm_pos, perm_neg, feature_names)
        perm_deltas.append(d)

    perm_deltas.sort()
    q = min(1.0, max(0.0, q))
    q_index = int(round(q * (len(perm_deltas) - 1)))
    bias = perm_deltas[q_index] if perm_deltas else 0.0
    delta_calibrated = max(0.0, observed - bias)
    w = max(0.0, min(1.0, construction_blend_weight))
    delta_reported = (1.0 - w) * delta_calibrated
    return delta_reported, bias, observed, used, per_feature, top_feature


def resolve_rmc_case_b_feature_set(name: str) -> Set[str]:
    """Return a named pre-registered Case-B feature set."""
    normalized = (name or "final").strip().lower()
    if normalized in {"final", "residual", "controlled"}:
        return set(RMC_CASEB_FINAL_FEATURES)
    if normalized in {"diagnostic", "all", "all_features"}:
        return set(RMC_CASEB_DIAGNOSTIC_FEATURES)
    if normalized in {"no_filesize", "no-file-size", "no_file_size"}:
        return set(RMC_CASEB_NO_FILESIZE_FEATURES)
    raise ValueError("unknown RMC Case-B feature set: {}".format(name))

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


def compute_rmc_metrics(
    instances: List[InstanceRecord],
    pairs: List[PairRecord],
    rmc_alpha: float = 0.05,
    delta_g: float = 0.0,
    estimate_delta_g: bool = False,
    case_b_mode: str = "final",
    case_b_feature_set: str = "final",
    case_b_permutations: int = 200,
    case_b_permutation_quantile: float = 0.95,
    case_b_seed: int = 0,
) -> Dict:
    """Compute RMC (Reasoning-Margin Certification) per Idea_Structure.tex §3–§4.

    Discrimination gate (§3.3 Route 1 — factorized LCB, fixes pseudo-replication):
        LCB_factorized(ŝ_G) = CP_LCB_{α/2}(θ_P) × CP_LCB_{α/2}(θ_N)
    where α' = rmc_alpha / 2 (Bonferroni split) so joint coverage ≥ 1 − rmc_alpha.

    Null ceiling (§3.5.2 Case A, δ_G given / assumed):
        C_G = ((1 + delta_g) / 2)^2  →  0.25 when delta_g = 0.

    Case B approximation:
        when estimate_delta_g=True, δ_G is estimated per group by a CNF
        side-channel battery and C_G is recomputed per group. For Case B we
        remove the Case-A 0.25 baseline offset, so the estimated ceiling is
        ((1 + δ_G) / 2)^2 - 0.25. The recommended final mode uses a
        pre-registered residual feature set plus permutation calibration;
        diagnostic mode reproduces the original all-feature max-KS estimator
        and is intended as a leakage audit rather than the final certification
        ceiling.

    Case C (feature-level residualization + construction-level blend):
        keeps the same ceiling formula as Case B, but estimates δ_G from
        complexity-debiased residual side-channel features.  For every
        feature f_k in the residual battery, a ridge regression
            f_k ~ 1 + (num_vars, num_clauses, num_literals, file_size_bytes)
        is fitted across all instances; per-instance residuals replace the
        raw feature values, and the same max-KS over residuals plus
        permutation calibration give a Frisch--Waugh--Lovell point estimate
        $\hat\delta_{FWL}$ of the residual distinguishability.  The reported
        Case C $\hat\delta_G$ is then a convex blend with the Case A
        construction-level reference $\delta=0$:
            $\hat\delta_G = (1 - w)\,\hat\delta_{FWL}$,
        with $w = $ ``RMC_CASEC_CONSTRUCTION_BLEND_WEIGHT`` (default 0.5).
        This positions Case C halfway between the FWL point estimate and the
        construction-level idealization in $\delta$-space.

    Case D (separability only):
        ignores shortcut ceilings and witness gating. The reported RMC is the
        factorized separability lower bound itself:
            RMC_D(G) = CP_LCB_{α/2}(θ_P) × CP_LCB_{α/2}(θ_N).
        Accordingly, the discrimination margin equals the factorized LCB and
        both RMC_signed(G) and RMC_clipped(G) collapse to that same value.

    Discrimination margin = LCB_factorized(ŝ_G) − C_G.

    Witness gate (§4.2 SAT hard certificate):
        LCB_{rmc_alpha}(w_G^+ / a_G)
    w_G^+ = SAT instances with a verified satisfying assignment (from PairRecord).
    a_G   = SAT instances with any valid prediction (from InstanceRecord).
    Witness null = 0 (Case A, §4.2); no UCB on ceiling needed for this gate.

    RMC_signed(G) = min(discrimination_margin, witness_margin)   [§4.2 min-gate]
    RMC_clipped(G) = max(0, RMC_signed(G))

    Aggregate metrics are macro-means over groups with sufficient data.
    """
    inst_by_group: Dict = defaultdict(list)
    for item in instances:
        inst_by_group[(item.n_value, item.group_id)].append(item)

    pair_by_group: Dict = defaultdict(list)
    for pair in pairs:
        pair_by_group[(pair.n_value, pair.group_id)].append(pair)

    fixed_null_ceiling = ((1.0 + delta_g) / 2.0) ** 2
    alpha_half = rmc_alpha / 2.0  # Bonferroni split for factorized LCB

    normalized_case_b_mode = (case_b_mode or "final").strip().lower()
    if normalized_case_b_mode not in {"final", "final_raw", "diagnostic", "case_c_final", "case_c_final_raw", "case_d"}:
        raise ValueError(
            "case_b_mode must be one of: final, final_raw, diagnostic, case_c_final, case_c_final_raw, case_d"
        )
    is_case_d_mode = normalized_case_b_mode == "case_d"
    final_feature_whitelist = resolve_rmc_case_b_feature_set(case_b_feature_set)
    diagnostic_feature_whitelist = resolve_rmc_case_b_feature_set("diagnostic")
    case_c_models: Dict[str, List[float]] = (
        fit_case_c_feature_residualization(instances, final_feature_whitelist)
        if estimate_delta_g and normalized_case_b_mode in {"case_c_final", "case_c_final_raw"}
        else {}
    )
    rng = random.Random(case_b_seed)

    group_rows = []
    all_keys = sorted(set(list(inst_by_group.keys()) + list(pair_by_group.keys())))

    for key in all_keys:
        n_value, gid = key
        items = inst_by_group.get(key, [])
        grp_pairs = pair_by_group.get(key, [])
        if is_case_d_mode:
            diagnostic_delta_g = None
            diagnostic_features_used = 0
            diagnostic_per_feature = {}
            diagnostic_top_feature = None
            diagnostic_null_ceiling = 0.0
            group_delta_g = 0.0
            delta_g_observed = 0.0
            delta_g_bias = 0.0
            side_channel_features_used = 0
            per_feature_ks = {}
            top_feature = None
        else:
            diagnostic_delta_g, diagnostic_features_used, diagnostic_per_feature, diagnostic_top_feature = (
                estimate_group_delta_g_detailed(items, feature_whitelist=diagnostic_feature_whitelist)
            )
            diagnostic_null_ceiling = ((1.0 + diagnostic_delta_g) / 2.0) ** 2 - 0.25

        if estimate_delta_g and not is_case_d_mode:
            if normalized_case_b_mode == "diagnostic":
                group_delta_g = diagnostic_delta_g
                delta_g_observed = diagnostic_delta_g
                delta_g_bias = 0.0
                side_channel_features_used = diagnostic_features_used
                per_feature_ks = diagnostic_per_feature
                top_feature = diagnostic_top_feature
            elif normalized_case_b_mode == "final_raw":
                group_delta_g, side_channel_features_used, per_feature_ks, top_feature = (
                    estimate_group_delta_g_detailed(items, feature_whitelist=final_feature_whitelist)
                )
                delta_g_observed = group_delta_g
                delta_g_bias = 0.0
            elif normalized_case_b_mode == "case_c_final_raw":
                group_delta_g, side_channel_features_used, per_feature_ks, top_feature = (
                    estimate_group_delta_g_case_c_detailed(
                        items,
                        residual_models=case_c_models,
                        feature_whitelist=final_feature_whitelist,
                    )
                )
                delta_g_observed = group_delta_g
                delta_g_bias = 0.0
            else:
                if normalized_case_b_mode == "case_c_final":
                    estimator_output = estimate_group_delta_g_case_c_permutation_calibrated(
                        items,
                        residual_models=case_c_models,
                        feature_whitelist=final_feature_whitelist,
                        num_perm=case_b_permutations,
                        q=case_b_permutation_quantile,
                        rng=rng,
                    )
                else:
                    estimator_output = estimate_group_delta_g_permutation_calibrated(
                        items,
                        feature_whitelist=final_feature_whitelist,
                        num_perm=case_b_permutations,
                        q=case_b_permutation_quantile,
                        rng=rng,
                    )
                (
                    group_delta_g,
                    delta_g_bias,
                    delta_g_observed,
                    side_channel_features_used,
                    per_feature_ks,
                    top_feature,
                ) = estimator_output
        elif not is_case_d_mode:
            group_delta_g = delta_g
            delta_g_observed = delta_g
            delta_g_bias = 0.0
            side_channel_features_used = 0
            per_feature_ks = {}
            top_feature = None
        null_ceiling = (
            0.0
            if is_case_d_mode
            else (
                ((1.0 + group_delta_g) / 2.0) ** 2 - 0.25
                if estimate_delta_g
                else fixed_null_ceiling
            )
        )

        sat_eval = [x for x in items if x.true_label == SAT and x.pred_label in {SAT, UNSAT}]
        unsat_eval = [x for x in items if x.true_label == UNSAT and x.pred_label in {SAT, UNSAT}]
        num_sat_eval = len(sat_eval)
        num_unsat_eval = len(unsat_eval)

        p_plus = sum(1 for x in sat_eval if x.pred_label == SAT)
        n_minus = sum(1 for x in unsat_eval if x.pred_label == UNSAT)

        # θ̂_P and θ̂_N (§3.3)
        theta_P_hat = _safe_div(p_plus, num_sat_eval)
        theta_N_hat = _safe_div(n_minus, num_unsat_eval)
        s_hat = (
            theta_P_hat * theta_N_hat
            if theta_P_hat is not None and theta_N_hat is not None
            else None
        )

        # Factorized LCB — fixes the pseudo-replication bug (§3.3 §6.2)
        lcb_theta_P = (
            clopper_pearson_lcb(p_plus, num_sat_eval, alpha=alpha_half)
            if num_sat_eval > 0
            else None
        )
        lcb_theta_N = (
            clopper_pearson_lcb(n_minus, num_unsat_eval, alpha=alpha_half)
            if num_unsat_eval > 0
            else None
        )
        lcb_s_factorized = (
            lcb_theta_P * lcb_theta_N
            if lcb_theta_P is not None and lcb_theta_N is not None
            else None
        )

        disc_margin = (
            lcb_s_factorized - null_ceiling if lcb_s_factorized is not None else None
        )

        # Witness margin: w_G^+ from PairRecord.sat_assignment_satisfied (already CDCL-verified)
        # a_G = SAT instances with valid predictions (denominator for witness rate)
        a_G = num_sat_eval
        w_G_plus = sum(1 for p in grp_pairs if p.sat_assignment_satisfied)
        witness_rate = _safe_div(w_G_plus, a_G)
        # Witness null = 0 (Case A), use full rmc_alpha budget (§4.2)
        lcb_witness = (
            clopper_pearson_lcb(w_G_plus, a_G, alpha=rmc_alpha) if a_G > 0 else None
        )

        # Two-gate min (§4.2): both discrimination and witness must be cleared
        if is_case_d_mode:
            rmc_signed = lcb_s_factorized
        elif disc_margin is not None and lcb_witness is not None:
            rmc_signed = min(disc_margin, lcb_witness)
        else:
            rmc_signed = None
        rmc_clipped = (
            rmc_signed
            if is_case_d_mode
            else (max(0.0, rmc_signed) if rmc_signed is not None else None)
        )

        group_rows.append({
            "N": n_value,
            "group_id": gid,
            "num_sat_eval": num_sat_eval,
            "num_unsat_eval": num_unsat_eval,
            "p_plus": p_plus,
            "n_minus": n_minus,
            "theta_P_hat": theta_P_hat,
            "theta_N_hat": theta_N_hat,
            "s_hat": s_hat,
            "lcb_theta_P": lcb_theta_P,
            "lcb_theta_N": lcb_theta_N,
            "lcb_s_factorized": lcb_s_factorized,
            "delta_g": group_delta_g,
            "delta_g_observed": delta_g_observed,
            "delta_g_bias": delta_g_bias,
            "diagnostic_delta_g": diagnostic_delta_g,
            "diagnostic_null_ceiling": diagnostic_null_ceiling,
            "side_channel_features_used": side_channel_features_used,
            "side_channel_top_feature": top_feature,
            "side_channel_top_feature_diagnostic": diagnostic_top_feature,
            "side_channel_ks_by_feature": per_feature_ks,
            "side_channel_ks_by_feature_diagnostic": diagnostic_per_feature,
            "null_ceiling": null_ceiling,
            "disc_margin": disc_margin,
            "a_G": a_G,
            "w_G_plus": w_G_plus,
            "witness_rate": witness_rate,
            "lcb_witness": lcb_witness,
            "rmc_signed": rmc_signed,
            "rmc_clipped": rmc_clipped,
        })

    valid_rows = [r for r in group_rows if r["lcb_s_factorized"] is not None]

    def _mean_valid(vals: list) -> Optional[float]:
        clean = [v for v in vals if v is not None]
        return sum(clean) / len(clean) if clean else None

    signed_vals = [r["rmc_signed"] for r in valid_rows]
    certified_count = sum(1 for v in signed_vals if v is not None and v > 0)
    delta_vals = [r["delta_g"] for r in valid_rows]
    ceiling_vals = [r["null_ceiling"] for r in valid_rows]
    observed_delta_vals = [r["delta_g_observed"] for r in valid_rows]
    delta_bias_vals = [r["delta_g_bias"] for r in valid_rows]
    diagnostic_delta_vals = [r["diagnostic_delta_g"] for r in valid_rows]
    diagnostic_ceiling_vals = [r["diagnostic_null_ceiling"] for r in valid_rows]
    features_used_vals = [r["side_channel_features_used"] for r in valid_rows]

    def _mode_string(vals: list) -> Optional[str]:
        clean = [v for v in vals if v]
        if not clean:
            return None
        counts = defaultdict(int)
        for v in clean:
            counts[v] += 1
        return max(counts, key=lambda k: (counts[k], k))

    if is_case_d_mode:
        reported_case_b_mode = "case_d"
        reported_feature_set = "separability_only"
    elif not estimate_delta_g:
        reported_case_b_mode = "fixed"
        reported_feature_set = "fixed"
    else:
        reported_case_b_mode = normalized_case_b_mode
        reported_feature_set = case_b_feature_set

    return {
        "RMC_delta_g": 0.0 if is_case_d_mode else (_mean_valid(delta_vals) if estimate_delta_g else delta_g),
        "RMC_null_ceiling": 0.0 if is_case_d_mode else (_mean_valid(ceiling_vals) if estimate_delta_g else fixed_null_ceiling),
        "RMC_delta_g_max": 0.0 if is_case_d_mode else (max(delta_vals) if delta_vals else None),
        "RMC_null_ceiling_max": 0.0 if is_case_d_mode else (max(ceiling_vals) if ceiling_vals else None),
        "RMC_delta_source": "separability_only_case_d" if is_case_d_mode else ("estimated_cnf_side_channel" if estimate_delta_g else "fixed"),
        "RMC_case_b_mode": reported_case_b_mode,
        "RMC_case_b_feature_set": reported_feature_set,
        "RMC_delta_g_observed": 0.0 if is_case_d_mode else _mean_valid(observed_delta_vals),
        "RMC_delta_g_bias": 0.0 if is_case_d_mode else _mean_valid(delta_bias_vals),
        "RMC_delta_g_diagnostic": _mean_valid(diagnostic_delta_vals),
        "RMC_null_ceiling_diagnostic": _mean_valid(diagnostic_ceiling_vals),
        "RMC_side_channel_features_used": _mean_valid(features_used_vals),
        "RMC_side_channel_top_feature": _mode_string([r["side_channel_top_feature"] for r in valid_rows]),
        "RMC_side_channel_top_feature_diagnostic": _mode_string([r["side_channel_top_feature_diagnostic"] for r in valid_rows]),
        "RMC_theta_P_hat": _mean_valid([r["theta_P_hat"] for r in valid_rows]),
        "RMC_theta_N_hat": _mean_valid([r["theta_N_hat"] for r in valid_rows]),
        "RMC_s_hat": _mean_valid([r["s_hat"] for r in valid_rows]),
        "RMC_lcb_theta_P": _mean_valid([r["lcb_theta_P"] for r in valid_rows]),
        "RMC_lcb_theta_N": _mean_valid([r["lcb_theta_N"] for r in valid_rows]),
        "RMC_lcb_s_factorized": _mean_valid([r["lcb_s_factorized"] for r in valid_rows]),
        "RMC_discrimination_margin": _mean_valid([r["disc_margin"] for r in valid_rows]),
        "RMC_witness_rate": _mean_valid([r["witness_rate"] for r in valid_rows]),
        "RMC_lcb_witness": _mean_valid([r["lcb_witness"] for r in valid_rows]),
        "RMC_signed": _mean_valid(signed_vals),
        "RMC_clipped": _mean_valid([r["rmc_clipped"] for r in valid_rows]),
        "RMC_num_groups_eval": len(valid_rows),
        "RMC_certified_fraction": _safe_div(certified_count, len(valid_rows)),
        "rmc_group_rows": group_rows,  # kept for JSON report; not written to CSV
    }


def compute_lcb_group_adr_metrics(
    instances: List[InstanceRecord],
    wilson_z: float = 1.96,
    cp_alpha: float = 0.05,
) -> Dict:
    """Compute pairing-invariant group ADR using Wilson and Clopper-Pearson LCB.

    For each cluster G=(P,N), where P are true SAT instances and N are true
    UNSAT instances, use all cross-label SAT/UNSAT comparisons rather than
    the original pair IDs.

    Let p_plus be the number of true SAT instances predicted SAT, and n_minus
    be the number of true UNSAT instances predicted UNSAT. Then

        k_G = p_plus * n_minus
        m_G = |P_eval| * |N_eval|

    where P_eval and N_eval include only instances with valid predictions.
    The raw group separability is k_G/m_G. In this script it is exported
    explicitly as GROUP_SEPARABILITY. The Wilson and Clopper-Pearson lower
    bounds are exported as conservative variants of the same cross-label
    separability statistic.
    """
    grouped = defaultdict(list)
    for item in instances:
        grouped[(item.n_value, item.group_id)].append(item)

    group_rows = []
    for (n_value, gid) in sorted(grouped):
        items = grouped[(n_value, gid)]

        sat_all = [x for x in items if x.true_label == SAT]
        unsat_all = [x for x in items if x.true_label == UNSAT]
        sat_eval = [x for x in sat_all if x.pred_label in {SAT, UNSAT}]
        unsat_eval = [x for x in unsat_all if x.pred_label in {SAT, UNSAT}]

        p_plus = sum(1 for x in sat_eval if x.pred_label == SAT)
        n_minus = sum(1 for x in unsat_eval if x.pred_label == UNSAT)

        m_g = len(sat_eval) * len(unsat_eval)
        k_g = p_plus * n_minus
        raw_g = _safe_div(k_g, m_g) if m_g > 0 else None
        lcb_g = wilson_lcb(k_g, m_g, z=wilson_z) if m_g > 0 else None
        cp_lcb_g = clopper_pearson_lcb(k_g, m_g, alpha=cp_alpha) if m_g > 0 else None

        group_rows.append(
            {
                "N": n_value,
                "group_id": gid,
                "num_sat_all": len(sat_all),
                "num_unsat_all": len(unsat_all),
                "num_sat_eval": len(sat_eval),
                "num_unsat_eval": len(unsat_eval),
                "num_sat_pred_sat": p_plus,
                "num_unsat_pred_unsat": n_minus,
                "cross_success_count": k_g,
                "cross_comparison_count": m_g,
                "cross_separability_without_LCB": raw_g,
                "cross_separability_wilson_lcb": lcb_g,
                "cross_separability_cp_lcb": cp_lcb_g,
            }
        )

    valid_rows = [x for x in group_rows if x["cross_separability_wilson_lcb"] is not None]
    total_k = sum(x["cross_success_count"] for x in valid_rows)
    total_m = sum(x["cross_comparison_count"] for x in valid_rows)

    lcb_values = [x["cross_separability_wilson_lcb"] for x in valid_rows]
    cp_lcb_values = [x["cross_separability_cp_lcb"] for x in valid_rows]
    raw_values = [x["cross_separability_without_LCB"] for x in valid_rows if x["cross_separability_without_LCB"] is not None]

    mean_sat_eval = (
        _safe_div(sum(x["num_sat_eval"] for x in valid_rows), len(valid_rows)) if valid_rows else None
    )
    mean_unsat_eval = (
        _safe_div(sum(x["num_unsat_eval"] for x in valid_rows), len(valid_rows)) if valid_rows else None
    )

    raw_macro = _safe_div(sum(raw_values), len(raw_values)) if raw_values else None
    raw_micro = _safe_div(total_k, total_m) if total_m > 0 else None

    return {
        "GROUP_SEPARABILITY": raw_macro,
        "GROUP_SEPARABILITY_MICRO": raw_micro,
        "GROUP_SEPARABILITY_NUM_GROUPS_TOTAL": len(group_rows),
        "GROUP_SEPARABILITY_NUM_GROUPS_EVAL": len(valid_rows),
        "GROUP_SEPARABILITY_TOTAL_CROSS_SUCCESS": total_k,
        "GROUP_SEPARABILITY_TOTAL_CROSS_COMPARISONS": total_m,
        "GROUP_SEPARABILITY_MEAN_NUM_SAT_EVAL_PER_GROUP": mean_sat_eval,
        "GROUP_SEPARABILITY_MEAN_NUM_UNSAT_EVAL_PER_GROUP": mean_unsat_eval,
        "GROUP_SEPARABILITY_CHANCE_BASELINE": 0.25,

        "LCB_GADR": _safe_div(sum(lcb_values), len(lcb_values)) if lcb_values else None,
        "LCB_GADR_without_LCB": raw_macro,
        "LCB_GADR_micro": wilson_lcb(total_k, total_m, z=wilson_z) if total_m > 0 else None,
        "LCB_GADR_micro_without_LCB": raw_micro,
        "LCB_GADR_num_groups_total": len(group_rows),
        "LCB_GADR_num_groups_eval": len(valid_rows),
        "LCB_GADR_total_cross_success": total_k,
        "LCB_GADR_total_cross_comparisons": total_m,
        "LCB_GADR_mean_num_sat_eval_per_group": mean_sat_eval,
        "LCB_GADR_mean_num_unsat_eval_per_group": mean_unsat_eval,

        "CP_LCB_GADR": _safe_div(sum(cp_lcb_values), len(cp_lcb_values)) if cp_lcb_values else None,
        "CP_LCB_GADR_without_LCB": raw_macro,
        "CP_LCB_GADR_micro": clopper_pearson_lcb(total_k, total_m, alpha=cp_alpha) if total_m > 0 else None,
        "CP_LCB_GADR_micro_without_LCB": raw_micro,
        "CP_LCB_GADR_num_groups_total": len(group_rows),
        "CP_LCB_GADR_num_groups_eval": len(valid_rows),
        "CP_LCB_GADR_total_cross_success": total_k,
        "CP_LCB_GADR_total_cross_comparisons": total_m,
        "CP_LCB_GADR_mean_num_sat_eval_per_group": mean_sat_eval,
        "CP_LCB_GADR_mean_num_unsat_eval_per_group": mean_unsat_eval,

        "lcb_group_rows": group_rows,
    }


def compute_by_n(
    pairs: List[PairRecord],
    instances: List[InstanceRecord],
    sgadr_lambda: float = 1.0,
    wilson_z: float = 1.96,
    cp_alpha: float = 0.05,
    rmc_alpha: float = 0.05,
    rmc_delta_g: float = 0.0,
    estimate_delta_g: bool = False,
    case_b_mode: str = "final",
    case_b_feature_set: str = "final",
    case_b_permutations: int = 200,
    case_b_permutation_quantile: float = 0.95,
    case_b_seed: int = 0,
) -> List[Dict]:
    pairs_by_n = defaultdict(list)
    inst_by_n = defaultdict(list)
    for pair in pairs:
        pairs_by_n[pair.n_value].append(pair)
    for item in instances:
        inst_by_n[item.n_value].append(item)

    rows = []
    for n_value in sorted(set(list(pairs_by_n.keys()) + list(inst_by_n.keys()))):
        n_pairs = pairs_by_n.get(n_value, [])
        n_insts = inst_by_n.get(n_value, [])
        pair_metrics = compute_group_adr_metrics(n_pairs, sgadr_lambda=sgadr_lambda)
        lcb_metrics = compute_lcb_group_adr_metrics(n_insts, wilson_z=wilson_z, cp_alpha=cp_alpha)
        inst_metrics = compute_instance_metrics(n_insts)
        rmc_metrics = compute_rmc_metrics(
            n_insts,
            n_pairs,
            rmc_alpha=rmc_alpha,
            delta_g=rmc_delta_g,
            estimate_delta_g=estimate_delta_g,
            case_b_mode=case_b_mode,
            case_b_feature_set=case_b_feature_set,
            case_b_permutations=case_b_permutations,
            case_b_permutation_quantile=case_b_permutation_quantile,
            case_b_seed=case_b_seed + int(n_value),
        )
        rmc_metrics.pop("rmc_group_rows", None)
        metrics = add_derived_adr_metrics(
            dict(pair_metrics, **lcb_metrics, **inst_metrics, **rmc_metrics)
        )
        rows.append({"N": n_value, "metrics": metrics})
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
    rows: List[Dict] = []
    for row in by_n_rows:
        rows.append({key: row.get(key) for key in DIRECT_CLASSIFICATION_SUMMARY_HEADERS})
    return rows


def build_direct_classification_summary_overall_rows(overall_rows: List[Dict]) -> List[Dict]:
    rows: List[Dict] = []
    for row in overall_rows:
        rows.append({key: row.get(key) for key in DIRECT_CLASSIFICATION_SUMMARY_OVERALL_HEADERS})
    return rows


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
    parser.add_argument(
        "--output-prefix",
        default=DEFAULT_OUTPUT_PREFIX,
        help="Filename prefix for JSON/CSV/XLSX outputs.",
    )
    parser.add_argument("--preview-limit", type=int, default=40)
    parser.add_argument("--sgadr-lambda", type=float, default=1.0)
    parser.add_argument(
        "--wilson-z",
        type=float,
        default=1.96,
        help="z value used for Wilson lower confidence bound in LCB_GADR.",
    )
    parser.add_argument(
        "--cp-alpha",
        type=float,
        default=0.05,
        help="alpha used for one-sided Clopper-Pearson lower confidence bound in CP_LCB_GADR.",
    )
    parser.add_argument(
        "--rmc-alpha",
        type=float,
        default=0.05,
        help=(
            "Confidence level for RMC factorized LCB and witness LCB. "
            "Bonferroni split (alpha/2) is applied to each of θ_P and θ_N "
            "so the joint level is ≤ rmc_alpha. (Idea_Structure.tex §3.3 §4.1)"
        ),
    )
    parser.add_argument(
        "--rmc-delta-g",
        type=float,
        default=0.0,
        help=(
            "Assumed within-group side-channel distinguishability δ_G (Case A). "
            "With delta_g=0 the null ceiling equals 0.25. "
            "Set to a measured value when an adversary battery is available. "
            "(Idea_Structure.tex §3.5.2)"
        ),
    )
    parser.add_argument(
        "--estimate-delta-g",
        action="store_true",
        help=(
            "Estimate δ_G per group from a CNF side-channel feature battery "
            "instead of using the fixed --rmc-delta-g Case-A value."
        ),
    )
    parser.add_argument(
        "--rmc-case-b-mode",
        choices=["final", "final_raw", "diagnostic", "case_c_final", "case_c_final_raw", "case_d"],
        default="final",
        help=(
            "RMC scenario selector. "
            "final: residual feature set + permutation-calibrated excess KS; "
            "final_raw: residual feature set without calibration; "
            "diagnostic: original all-feature max-KS leakage audit; "
            "case_c_final: feature-level residualization against complexity covariates "
            "(per-feature ridge partial-out, Frisch--Waugh--Lovell) + permutation calibration, "
            "then blended toward the Case A construction-level reference (delta=0) at weight "
            "RMC_CASEC_CONSTRUCTION_BLEND_WEIGHT (default 0.5); "
            "case_c_final_raw: feature-level residualization without permutation calibration, "
            "same construction-level blend; "
            "case_d: use separability only and report the factorized LCB directly "
            "as RMC, with zero shortcut ceiling and no witness gate."
        ),
    )
    parser.add_argument(
        "--rmc-case-b-feature-set",
        choices=["final", "residual", "controlled", "diagnostic", "all", "all_features", "no_filesize"],
        default="final",
        help=(
            "Feature set for final/final_raw Case-B δ_G. The recommended final set "
            "uses residual shape features and excludes construction-size artifacts."
        ),
    )
    parser.add_argument(
        "--rmc-case-b-permutations",
        type=int,
        default=200,
        help="Number of within-group label permutations used for final Case-B calibration.",
    )
    parser.add_argument(
        "--rmc-case-b-permutation-quantile",
        type=float,
        default=0.95,
        help="Quantile of permuted max-KS used as the bias term in final Case-B calibration.",
    )
    parser.add_argument(
        "--rmc-case-b-seed",
        type=int,
        default=0,
        help="Random seed for final Case-B permutation calibration.",
    )
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
        overall_lcb = compute_lcb_group_adr_metrics(
            instances, wilson_z=args.wilson_z, cp_alpha=args.cp_alpha
        )
        overall_inst = compute_instance_metrics(instances)
        overall_rmc = compute_rmc_metrics(
            instances,
            pairs,
            rmc_alpha=args.rmc_alpha,
            delta_g=args.rmc_delta_g,
            estimate_delta_g=args.estimate_delta_g,
            case_b_mode=args.rmc_case_b_mode,
            case_b_feature_set=args.rmc_case_b_feature_set,
            case_b_permutations=args.rmc_case_b_permutations,
            case_b_permutation_quantile=args.rmc_case_b_permutation_quantile,
            case_b_seed=args.rmc_case_b_seed,
        )
        overall_rmc.pop("rmc_group_rows", None)
        overall = add_derived_adr_metrics(
            {**overall_pair, **overall_lcb, **overall_inst, **overall_rmc}
        )
        by_n = compute_by_n(
            pairs,
            instances,
            sgadr_lambda=args.sgadr_lambda,
            wilson_z=args.wilson_z,
            cp_alpha=args.cp_alpha,
            rmc_alpha=args.rmc_alpha,
            rmc_delta_g=args.rmc_delta_g,
            estimate_delta_g=args.estimate_delta_g,
            case_b_mode=args.rmc_case_b_mode,
            case_b_feature_set=args.rmc_case_b_feature_set,
            case_b_permutations=args.rmc_case_b_permutations,
            case_b_permutation_quantile=args.rmc_case_b_permutation_quantile,
            case_b_seed=args.rmc_case_b_seed,
        )
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

    output_prefix = args.output_prefix
    if args.rmc_case_b_mode == "case_d" and output_prefix == DEFAULT_OUTPUT_PREFIX:
        output_prefix = f"{DEFAULT_OUTPUT_PREFIX}_case_d"
    report_json = os.path.join(output_dir, f"{output_prefix}.json")
    with open(report_json, "w", encoding="utf-8") as f:
        json.dump(
            {
                "commercial_api_root": commercial_api_root,
                "input_roots_by_model": {model: input_roots for model, input_roots in roots},
                "sgadr_lambda": args.sgadr_lambda,
                "wilson_z": args.wilson_z,
                "cp_alpha": args.cp_alpha,
                "rmc_alpha": args.rmc_alpha,
                "rmc_delta_g": args.rmc_delta_g,
                "rmc_delta_source": (
                    "separability_only_case_d"
                    if args.rmc_case_b_mode == "case_d"
                    else ("estimated_cnf_side_channel" if args.estimate_delta_g else "fixed")
                ),
                "rmc_case_b_mode": (
                    "case_d"
                    if args.rmc_case_b_mode == "case_d"
                    else (args.rmc_case_b_mode if args.estimate_delta_g else "fixed")
                ),
                "rmc_case_b_feature_set": (
                    "separability_only"
                    if args.rmc_case_b_mode == "case_d"
                    else (args.rmc_case_b_feature_set if args.estimate_delta_g else "fixed")
                ),
                "rmc_case_b_permutations": (
                    0
                    if args.rmc_case_b_mode == "case_d"
                    else (args.rmc_case_b_permutations if args.estimate_delta_g else 0)
                ),
                "rmc_case_b_permutation_quantile": (
                    None
                    if args.rmc_case_b_mode == "case_d"
                    else (args.rmc_case_b_permutation_quantile if args.estimate_delta_g else None)
                ),
                "rmc_null_ceiling": (
                    0.0
                    if args.rmc_case_b_mode == "case_d"
                    else (None if args.estimate_delta_g else ((1.0 + args.rmc_delta_g) / 2.0) ** 2)
                ),
                "models": all_reports,
                "pairwise_preview_by_model": pairwise_preview_by_model,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    by_n_csv = os.path.join(output_dir, f"{output_prefix}.by_N.csv")
    write_csv(by_n_csv, BY_N_HEADERS, sorted(by_n_rows, key=lambda x: (str(x["model"]), int(x["N"]))))

    overall_csv = os.path.join(output_dir, f"{output_prefix}.overall.csv")
    write_csv(overall_csv, OVERALL_HEADERS, sorted(overall_rows, key=lambda x: str(x["model"])))

    direct_metric_by_n_rows = build_direct_classification_summary_rows(by_n_rows)
    direct_metric_overall_rows = build_direct_classification_summary_overall_rows(overall_rows)
    direct_metric_by_n_csv = os.path.join(
        output_dir, f"{output_prefix}.Existing_Category_Metrics.by_N.csv"
    )
    write_csv(
        direct_metric_by_n_csv,
        DIRECT_CLASSIFICATION_SUMMARY_HEADERS,
        sorted(direct_metric_by_n_rows, key=lambda x: (str(x["model"]), int(x["N"]))),
    )
    direct_metric_overall_csv = os.path.join(
        output_dir, f"{output_prefix}.Existing_Category_Metrics.overall.csv"
    )
    write_csv(
        direct_metric_overall_csv,
        DIRECT_CLASSIFICATION_SUMMARY_OVERALL_HEADERS,
        sorted(direct_metric_overall_rows, key=lambda x: str(x["model"])),
    )

    excel_path = os.path.join(output_dir, f"{output_prefix}.xlsx")
    excel_written, excel_error = write_excel(excel_path, by_n_rows, overall_rows)
    if not excel_written:
        print(f"[warn] excel export skipped: {excel_error}")

    direct_metric_excel_path = os.path.join(
        output_dir, f"{output_prefix}.Existing_Category_Metrics.xlsx"
    )
    direct_metric_excel_written, direct_metric_excel_error = write_direct_classification_summary_excel(
        direct_metric_excel_path,
        direct_metric_by_n_rows,
        direct_metric_overall_rows,
    )
    if not direct_metric_excel_written:
        print(f"[warn] direct classification metric excel export skipped: {direct_metric_excel_error}")

    preview_json = os.path.join(output_dir, f"{output_prefix}.pairwise_preview.json")
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
