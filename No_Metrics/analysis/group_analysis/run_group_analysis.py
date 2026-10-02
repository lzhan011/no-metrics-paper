#!/usr/bin/env python3
"""Run group-level metrics on the prediction archives used by the paper.

The domain adapters deliberately reuse each archive's established parsing and
deduplication code. Only the aggregation layer is new and domain-neutral.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import json
import platform
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from group_metrics import PairOutcome, compute_group_metrics


SCRIPT_DIR = Path(__file__).resolve().parent
WORKSPACE_ROOT = SCRIPT_DIR.parents[2]
SAT_ROOT = WORKSPACE_ROOT / "SAT_Group_Evaluation"
GRAPH_ROOT = WORKSPACE_ROOT / "Graph_Colouring_Evaluation"

COMMERCIAL_ANALYSIS_DIR = SAT_ROOT / "analysis" / "commercial_api_results_analysis"
TWO_SAT_ANALYSIS_DIR = SAT_ROOT / "analysis" / "open_source_LLM_2SAT_analysis"
GRAPH_ANALYSIS_DIR = SAT_ROOT / "analysis" / "graph_coloring_results_analysis"

COMMERCIAL_REPORT = COMMERCIAL_ANALYSIS_DIR / "commercial_api_group_adr_report_RMC_caseB.json"
TWO_SAT_REPORT = (
    TWO_SAT_ANALYSIS_DIR
    / "open_source_LLM_2SAT_group_adr_with_wilson_and_cp_RMC_rmc_paper_open_vendor_case_b.json"
)
GRAPH_REPORT = (
    GRAPH_ANALYSIS_DIR
    / "open_source_LLM_3_Colouring_group_adr_with_wilson_and_cp_RMC_graph_paper_case_a.json"
)
PRIMARY_LAMBDA = 1.0
DEFAULT_LAMBDAS = (0.0, 0.25, 0.5, 0.75, 1.0)
THRESHOLDS = tuple(range(1, 11))

THREE_SAT_SELECTION = {
    "GPT-5": "chatgpt__gpt-5",
    "Opus 4.7 (med. think)": (
        "claude__claude-opus-4-7__"
        "claude_batch_groups_zero_shot_desc_adaptive_thinking_medium_effort"
    ),
    "DeepSeek-Reasoner": "deepseek__deepseek-reasoner",
}


# Archived reports record the prediction inputs with the absolute paths of the
# machine they were produced on.  Everything the paper needs lives under the
# two evaluation trees, so any recorded path is re-rooted at this workspace:
#   <anything>/SAT_Group_Evaluation/<rest>      -> WORKSPACE_ROOT/SAT_Group_Evaluation/<rest>
#   <anything>/Graph_Colouring_Evaluation/<rest> -> WORKSPACE_ROOT/Graph_Colouring_Evaluation/<rest>
# A path that is already relative is resolved against WORKSPACE_ROOT.  This
# makes the analysis portable to a checkout of the repository.
_TREE_MARKERS = ("SAT_Group_Evaluation", "Graph_Colouring_Evaluation")


def rebase_path(path_text: object) -> Path:
    text = str(path_text).replace("\\", "/")
    for marker in _TREE_MARKERS:
        token = "/" + marker + "/"
        if token in text:
            return WORKSPACE_ROOT / marker / text.split(token, 1)[1]
        if text.startswith(marker + "/"):
            return WORKSPACE_ROOT / text
    candidate = Path(text)
    return candidate if candidate.is_absolute() else WORKSPACE_ROOT / candidate


def workspace_relative(path: Path) -> str:
    try:
        return str(Path(path).resolve().relative_to(WORKSPACE_ROOT))
    except ValueError:
        return str(path)


def _load_json(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _add_import_path(path: Path) -> None:
    text = str(path)
    if text not in sys.path:
        sys.path.insert(0, text)


def load_archive_modules():
    """Load the established per-domain parsers without modifying their files."""

    _add_import_path(COMMERCIAL_ANALYSIS_DIR)
    commercial = importlib.import_module("compute_commercial_api_metrics")

    _add_import_path(TWO_SAT_ANALYSIS_DIR)
    two_sat = importlib.import_module("compute_open_source_LLM_2SAT_metrics")

    _add_import_path(GRAPH_ANALYSIS_DIR)
    graph = importlib.import_module("graph_coloring_metrics_common")

    return commercial, two_sat, graph


WITNESS_STATUSES = (
    "satisfied",
    "unsatisfied",
    "missing_assignment",
    "invalid_assignment",
    "missing_cnf",
    "cdcl_unavailable",
)


def constraint_instance_outcomes_by_n(
    instances: Iterable[object],
    positive_label: str,
    negative_label: str,
    witness_verifier: Optional[Callable[[object], str]] = None,
    negative_witness_lookup: Optional[Callable[[object], Optional[bool]]] = None,
) -> Tuple[Dict[int, List[PairOutcome]], Dict[int, Dict[str, int]]]:
    """Pair deduplicated parsed instances and verify positive-side witnesses.

    Label-only metrics (GADR, ADR, SGADR, A>=k) use only the predicted labels.
    When ``witness_verifier`` is given it is the domain's established
    ``verify_assignment_for_instance`` (CDCL check of the SAT assignment for
    3-SAT/2-SAT, direct edge check of the coloring for graph coloring). It is
    called exactly as the archived pair builders do: only for a positive
    instance whose prediction is the positive label. The pair's
    ``positive_witness_verified`` flag is True iff the verifier returns
    ``"satisfied"``. ``negative_witness_lookup`` (2-SAT only) returns, for a
    negative instance predicted negative, whether the separate UNSAT follow-up
    produced a solver-verified unsatisfiable clause subset (True/False), or
    None when no follow-up record exists; this mirrors
    ``compute_open_source_LLM_2SAT_metrics.build_open_source_pairs``.  Per-``N`` counts of every verifier status are returned so
    the witness denominator can be audited.
    """

    buckets: Dict[Tuple[int, str, str], Dict[str, object]] = defaultdict(
        lambda: {"positive": None, "negative": None, "witness": False, "neg_witness": None}
    )
    status_by_n: Dict[int, Dict[str, int]] = defaultdict(
        lambda: {f"witness_status_{name}": 0 for name in WITNESS_STATUSES}
    )
    for instance in instances:
        n_value = int(getattr(instance, "n_value"))
        key = (
            n_value,
            str(getattr(instance, "group_id")),
            str(getattr(instance, "pair_id")),
        )
        true_label = getattr(instance, "true_label")
        prediction = getattr(instance, "pred_label")
        slot = "positive" if true_label == positive_label else "negative"
        if prediction is not None:
            buckets[key][slot] = prediction
            if (
                witness_verifier is not None
                and slot == "positive"
                and prediction == positive_label
            ):
                status = str(witness_verifier(instance))
                counter = status_by_n[n_value]
                counter[f"witness_status_{status}"] = (
                    counter.get(f"witness_status_{status}", 0) + 1
                )
                buckets[key]["witness"] = status == "satisfied"
            if (
                negative_witness_lookup is not None
                and slot == "negative"
                and prediction == negative_label
            ):
                value = negative_witness_lookup(instance)
                if value is not None:
                    buckets[key]["neg_witness"] = bool(value)

    by_n: Dict[int, List[PairOutcome]] = defaultdict(list)
    for (n_value, group_id, pair_id), values in sorted(buckets.items()):
        positive_pred = values["positive"]
        negative_pred = values["negative"]
        by_n[n_value].append(
            PairOutcome(
                group_id=group_id,
                pair_id=pair_id,
                positive_prediction_available=positive_pred is not None,
                negative_prediction_available=negative_pred is not None,
                positive_correct=positive_pred == positive_label,
                negative_correct=negative_pred == negative_label,
                positive_witness_verified=bool(values["witness"]),
                negative_witness_verified=values["neg_witness"],
            )
        )
    return by_n, status_by_n


def sum_status_counts(status_by_n: Mapping[int, Mapping[str, object]]) -> Dict[str, object]:
    total: Dict[str, object] = {f"witness_status_{name}": 0 for name in WITNESS_STATUSES}
    for counts in status_by_n.values():
        for key, value in counts.items():
            if key.startswith("witness_status_"):
                total[key] = int(total.get(key, 0)) + int(value)
            else:
                total[key] = value  # per-model provenance fields, identical across N
    return total


def alias_open_model(raw: str) -> str:
    if "Qwen3-14B" in raw:
        return "Qwen3-14B (think)"
    if "Qwen3-32B" in raw:
        return "Qwen3-32B (think)"
    if "QwQ-32B" in raw:
        return "QwQ-32B (think)"
    if "gpt-oss-20b" in raw:
        return "gpt-oss-20b (reason)"
    if "Nemotron-H-47B" in raw and "reasoning_on" in raw:
        return "Nemotron-H-47B (r-on)"
    if "Llama-3_3-Nemotron-Super-49B" in raw and "reasoning_on" in raw:
        return "Llama-Nemotron-Super-49B (r-on)"
    raise ValueError(f"Unrecognized paper model key: {raw}")


def archived_model_map(report: Mapping[str, object], key: str = "model") -> Dict[str, Dict]:
    return {str(row[key]): row for row in report.get("models", [])}


def archived_metrics_for_slice(model_row: Mapping[str, object], slice_type: str, slice_value: str) -> Optional[Dict]:
    if slice_type in {"overall", "scenario"}:
        value = model_row.get("overall_metrics")
        if isinstance(value, dict):
            return value
        value = model_row.get("adr_metrics")
        return value if isinstance(value, dict) else None
    if slice_type == "N":
        for row in model_row.get("metrics_by_N", []):
            if str(row.get("N")) == str(slice_value):
                value = row.get("metrics")
                return value if isinstance(value, dict) else None
    return None


def add_metric_slice(
    *,
    domain: str,
    model: str,
    configuration: str,
    slice_type: str,
    slice_value: str,
    pairs: Sequence[PairOutcome],
    lambdas: Sequence[float],
    archived_metrics: Optional[Mapping[str, object]],
    source_id: str,
    rows: List[Dict[str, object]],
    group_rows: List[Dict[str, object]],
    validation_rows: List[Dict[str, object]],
    extra_fields: Optional[Mapping[str, object]] = None,
) -> None:
    common = {
        "domain": domain,
        "model": model,
        "configuration": configuration,
        "slice_type": slice_type,
        "slice_value": slice_value,
        "source_id": source_id,
    }
    for lambda_value in lambdas:
        metrics, groups = compute_group_metrics(
            pairs,
            sgadr_lambda=lambda_value,
            thresholds=THRESHOLDS,
        )
        row = {**common, **metrics, **dict(extra_fields or {})}
        max_pairs = int(metrics["max_pairs_in_group"])
        row["group_concept_applicable"] = bool(metrics["num_groups_total"])
        row["threshold_10_structurally_possible"] = max_pairs >= 10
        row["threshold_10_interpretation"] = (
            "substantive repeated-success threshold"
            if max_pairs >= 10
            else "mechanical zero: fewer than 10 candidate pairs per group"
        )
        rows.append(row)

        if abs(lambda_value - PRIMARY_LAMBDA) < 1e-12:
            for group in groups:
                group_rows.append({**common, **group})
            if archived_metrics is not None:
                keys = ["GADR", "ADR", "SGADR", "SGADR_star"] + [
                    f"group_level_ADR_at_least_{threshold}_pair_success"
                    for threshold in THRESHOLDS
                ]
                # Witness-aware metrics are stored under ``*_with_assignment``
                # in the archived reports (SAT assignment / coloring assignment).
                witness_key_map = {
                    "GADR_with_witness": "GADR_with_assignment",
                    "ADR_with_witness": "ADR_with_assignment",
                    "SGADR_with_witness": "SGADR_with_assignment",
                    "SGADR_star_with_witness": "SGADR_star_with_assignment",
                    "pair_success_with_witness_total": "pair_success_with_assignment_total",
                    "GADR_with_two_sided_witness": "GADR_with_assignment_and_unsat_subset",
                    "ADR_with_two_sided_witness": "ADR_with_assignment_and_unsat_subset",
                    "SGADR_with_two_sided_witness": "SGADR_with_assignment_and_unsat_subset",
                    "SGADR_star_with_two_sided_witness": "SGADR_star_with_assignment_and_unsat_subset",
                    "pair_success_with_two_sided_witness_total": "pair_success_with_assignment_and_unsat_subset_total",
                }
                comparisons = [(key, key) for key in keys] + list(witness_key_map.items())
                for key, archived_key in comparisons:
                    if archived_key not in archived_metrics:
                        continue
                    old = archived_metrics.get(archived_key)
                    new = metrics.get(key)
                    if new is None and "two_sided" in key:
                        # No follow-up records for this slice: the archive
                        # stores 0.0 here, we store None; not a reproduction
                        # failure, so skip rather than compare.
                        continue
                    difference = (
                        abs(float(new) - float(old))
                        if new is not None and old is not None
                        else None
                    )
                    validation_rows.append(
                        {
                            **common,
                            "metric": key,
                            "archived_metric": archived_key,
                            "new_value": new,
                            "archived_value": old,
                            "absolute_difference": difference,
                            "matches_within_1e-12": (
                                difference is not None and difference <= 1e-12
                            ),
                        }
                    )


def run_three_sat(
    commercial,
    report: Dict,
    lambdas: Sequence[float],
    rows: List[Dict[str, object]],
    group_rows: List[Dict[str, object]],
    validation_rows: List[Dict[str, object]],
    input_paths: List[Path],
) -> None:
    report_models = archived_model_map(report)
    roots_by_model = report["input_roots_by_model"]
    commercial.MIN_EDIT_DATASET_DIR = "min_edit_3sat_dataset"
    for alias, report_key in THREE_SAT_SELECTION.items():
        print(f"[3-SAT] loading {alias}", flush=True)
        roots = [rebase_path(path) for path in roots_by_model[report_key]]
        input_paths.extend(roots)
        instances, _ = commercial.load_instances_from_roots([str(path) for path in roots])
        by_n, status_by_n = constraint_instance_outcomes_by_n(
            instances,
            commercial.SAT,
            commercial.UNSAT,
            witness_verifier=commercial.verify_assignment_for_instance,
        )
        all_outcomes = [outcome for n_value in sorted(by_n) for outcome in by_n[n_value]]
        archived_row = report_models[report_key]
        add_metric_slice(
            domain="3-SAT",
            model=alias,
            configuration=report_key,
            slice_type="overall",
            slice_value="all",
            pairs=all_outcomes,
            lambdas=lambdas,
            archived_metrics=archived_metrics_for_slice(archived_row, "overall", "all"),
            source_id=workspace_relative(COMMERCIAL_REPORT),
            rows=rows,
            group_rows=group_rows,
            validation_rows=validation_rows,
            extra_fields=sum_status_counts(status_by_n),
        )
        for n_value in sorted(by_n):
            add_metric_slice(
                domain="3-SAT",
                model=alias,
                configuration=report_key,
                slice_type="N",
                slice_value=str(n_value),
                pairs=by_n[n_value],
                lambdas=lambdas,
                archived_metrics=archived_metrics_for_slice(archived_row, "N", str(n_value)),
                source_id=workspace_relative(COMMERCIAL_REPORT),
                rows=rows,
                group_rows=group_rows,
                validation_rows=validation_rows,
                extra_fields=status_by_n.get(n_value, {}),
            )
        print(f"[3-SAT] finished {alias}: {len(all_outcomes)} candidate pairs", flush=True)


def run_two_sat(
    commercial,
    two_sat,
    report: Dict,
    lambdas: Sequence[float],
    rows: List[Dict[str, object]],
    group_rows: List[Dict[str, object]],
    validation_rows: List[Dict[str, object]],
    input_paths: List[Path],
) -> None:
    report_models = archived_model_map(report)
    commercial.MIN_EDIT_DATASET_DIR = "min_edit_2sat_dataset"
    for report_key, input_path_text in sorted(report["input_results_by_model"].items()):
        alias = alias_open_model(report_key)
        print(f"[2-SAT] loading {alias}", flush=True)
        input_path = rebase_path(input_path_text)
        input_paths.append(input_path)
        instances, _ = two_sat.load_instances_from_results(str(input_path))
        # Negative-side witness: the archived report records, per model, the
        # UNSAT clause-subset follow-up results file (None when that model was
        # never run through the follow-up). Records are keyed by the dataset-
        # relative CNF path, exactly as build_open_source_pairs does.
        followup_text = archived_model_map(report)[report_key].get(
            "unsat_clause_subset_followup_results"
        )
        followup_path = rebase_path(followup_text) if isinstance(followup_text, str) else None
        negative_lookup = None
        followup_stats: Dict[str, object] = {"negative_witness_followup_results": None}
        if followup_path is not None and followup_path.is_file():
            input_paths.append(followup_path)
            unsat_subset_by_key, subset_stats = two_sat.load_unsat_clause_subset_results(
                str(followup_path)
            )
            followup_stats = {
                "negative_witness_followup_results": workspace_relative(followup_path),
                "negative_witness_followup_records": subset_stats.get("accepted_followup_records"),
                "negative_witness_followup_success_records": subset_stats.get("followup_success_records"),
            }

            def negative_lookup(item, _map=unsat_subset_by_key):
                return _map.get(two_sat.canonical_instance_dedup_key(item))

        # The archived 2-SAT builder verifies SAT assignments with the shared
        # commercial CDCL checker (compute_open_source_LLM_2SAT_metrics.build_open_source_pairs).
        by_n, status_by_n = constraint_instance_outcomes_by_n(
            instances,
            commercial.SAT,
            commercial.UNSAT,
            witness_verifier=commercial.verify_assignment_for_instance,
            negative_witness_lookup=negative_lookup,
        )
        for counts in status_by_n.values():
            counts.update(followup_stats)
        all_outcomes = [outcome for n_value in sorted(by_n) for outcome in by_n[n_value]]
        archived_row = report_models[report_key]
        add_metric_slice(
            domain="2-SAT",
            model=alias,
            configuration=report_key,
            slice_type="overall",
            slice_value="all",
            pairs=all_outcomes,
            lambdas=lambdas,
            archived_metrics=archived_metrics_for_slice(archived_row, "overall", "all"),
            source_id=workspace_relative(TWO_SAT_REPORT),
            rows=rows,
            group_rows=group_rows,
            validation_rows=validation_rows,
            extra_fields=sum_status_counts(status_by_n),
        )
        for n_value in sorted(by_n):
            add_metric_slice(
                domain="2-SAT",
                model=alias,
                configuration=report_key,
                slice_type="N",
                slice_value=str(n_value),
                pairs=by_n[n_value],
                lambdas=lambdas,
                archived_metrics=archived_metrics_for_slice(archived_row, "N", str(n_value)),
                source_id=workspace_relative(TWO_SAT_REPORT),
                rows=rows,
                group_rows=group_rows,
                validation_rows=validation_rows,
                extra_fields=status_by_n.get(n_value, {}),
            )
        print(f"[2-SAT] finished {alias}: {len(all_outcomes)} candidate pairs", flush=True)


def run_graph_coloring(
    commercial,
    graph,
    report: Dict,
    lambdas: Sequence[float],
    rows: List[Dict[str, object]],
    group_rows: List[Dict[str, object]],
    validation_rows: List[Dict[str, object]],
    input_paths: List[Path],
) -> None:
    report_models = archived_model_map(report)
    commercial.MIN_EDIT_DATASET_DIR = "min_edit_graph_coloring_dataset"
    selected = {}
    for report_key, path in report["input_results_by_model"].items():
        try:
            selected[alias_open_model(report_key)] = (report_key, rebase_path(path))
        except ValueError:
            continue
    if len(selected) != 6:
        raise ValueError(f"Expected six graph paper configurations, found {sorted(selected)}")

    for alias, (report_key, input_path) in sorted(selected.items()):
        print(f"[Graph coloring] loading {alias}", flush=True)
        input_paths.append(input_path)
        instances, _ = graph.load_instances_from_results(str(input_path))
        by_n, status_by_n = constraint_instance_outcomes_by_n(
            instances,
            commercial.SAT,
            commercial.UNSAT,
            witness_verifier=graph.verify_assignment_for_instance,
        )
        all_outcomes = [outcome for n_value in sorted(by_n) for outcome in by_n[n_value]]
        archived_row = report_models[report_key]
        add_metric_slice(
            domain="Graph 3-coloring",
            model=alias,
            configuration=report_key,
            slice_type="overall",
            slice_value="all",
            pairs=all_outcomes,
            lambdas=lambdas,
            archived_metrics=archived_metrics_for_slice(archived_row, "overall", "all"),
            source_id=workspace_relative(GRAPH_REPORT),
            rows=rows,
            group_rows=group_rows,
            validation_rows=validation_rows,
            extra_fields=sum_status_counts(status_by_n),
        )
        for n_value in sorted(by_n):
            add_metric_slice(
                domain="Graph 3-coloring",
                model=alias,
                configuration=report_key,
                slice_type="N",
                slice_value=str(n_value),
                pairs=by_n[n_value],
                lambdas=lambdas,
                archived_metrics=archived_metrics_for_slice(archived_row, "N", str(n_value)),
                source_id=workspace_relative(GRAPH_REPORT),
                rows=rows,
                group_rows=group_rows,
                validation_rows=validation_rows,
                extra_fields=status_by_n.get(n_value, {}),
            )
        print(
            f"[Graph coloring] finished {alias}: {len(all_outcomes)} candidate pairs",
            flush=True,
        )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fingerprint_path(path: Path) -> Dict[str, object]:
    if path.is_file():
        return {
            "path": workspace_relative(path),
            "kind": "file",
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
    if path.is_dir():
        files = sorted(candidate for candidate in path.rglob("*.cnf.json") if candidate.is_file())
        digest = hashlib.sha256()
        total_bytes = 0
        for candidate in files:
            relative = str(candidate.relative_to(path)).encode("utf-8")
            digest.update(relative)
            digest.update(b"\0")
            with candidate.open("rb") as handle:
                for block in iter(lambda: handle.read(1024 * 1024), b""):
                    total_bytes += len(block)
                    digest.update(block)
        return {
            "path": str(path),
            "kind": "directory",
            "file_pattern": "*.cnf.json",
            "file_count": len(files),
            "size_bytes": total_bytes,
            "aggregate_sha256": digest.hexdigest(),
        }
    return {"path": str(path), "kind": "missing"}


def write_csv(path: Path, rows: Sequence[Mapping[str, object]], leading_fields: Sequence[str] = ()) -> None:
    all_fields = set()
    for row in rows:
        all_fields.update(row.keys())
    fields = list(leading_fields) + sorted(all_fields - set(leading_fields))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def build_applicability_rows(primary_rows: Sequence[Mapping[str, object]]) -> List[Dict[str, object]]:
    result = []
    domain_notes = {
        "3-SAT": "Natural generator group; up to 11 matched SAT/UNSAT pairs.",
        "2-SAT": "Natural generator group; up to 11 matched SAT/UNSAT pairs.",
        "Graph 3-coloring": "Natural generator group; up to 11 matched colorable/non-colorable pairs.",
    }
    for domain in sorted({str(row["domain"]) for row in primary_rows}):
        selected = [row for row in primary_rows if row["domain"] == domain]
        result.append(
            {
                "domain": domain,
                "group_concept_applicable": all(bool(row["group_concept_applicable"]) for row in selected),
                "num_primary_slices": len(selected),
                "min_pairs_per_group_maximum": min(int(row["max_pairs_in_group"]) for row in selected),
                "max_pairs_per_group_maximum": max(int(row["max_pairs_in_group"]) for row in selected),
                "num_slices_where_threshold_10_is_structurally_possible": sum(
                    bool(row["threshold_10_structurally_possible"]) for row in selected
                ),
                "interpretation": domain_notes[domain],
            }
        )
    return result


def write_results_markdown(
    path: Path,
    rows: Sequence[Mapping[str, object]],
    primary_rows: Sequence[Mapping[str, object]],
    validation_rows: Sequence[Mapping[str, object]],
    applicability_rows: Sequence[Mapping[str, object]],
) -> None:
    by_domain = defaultdict(list)
    for row in primary_rows:
        by_domain[str(row["domain"])].append(row)
    validation_differences = [
        float(row["absolute_difference"])
        for row in validation_rows
        if row.get("absolute_difference") is not None
    ]
    failed_validations = [
        row for row in validation_rows if not bool(row["matches_within_1e-12"])
    ]
    lines = [
        "# Group-analysis results",
        "",
        "## Run scope",
        "",
        f"- Lambda settings: {', '.join(str(x) for x in sorted({row['sgadr_lambda'] for row in rows}))}",
        f"- Primary lambda: {PRIMARY_LAMBDA}",
        f"- Primary metric slices: {len(primary_rows)}",
        f"- Archived-value comparisons: {len(validation_rows)}",
        f"- Archived-value mismatches: {len(failed_validations)}",
        f"- Maximum absolute reproduction difference: {max(validation_differences, default=0.0):.3g}",
        "",
        "## Domain coverage",
        "",
        "| Domain | Primary slices | Models/configurations | Threshold-10 structurally possible |",
        "|---|---:|---:|---:|",
    ]
    for domain in sorted(by_domain):
        domain_rows = by_domain[domain]
        lines.append(
            "| {} | {} | {} | {} |".format(
                domain,
                len(domain_rows),
                len({(row["model"], row["configuration"]) for row in domain_rows}),
                sum(bool(row["threshold_10_structurally_possible"]) for row in domain_rows),
            )
        )
    lines.extend(["", "## Applicability", ""])
    for item in applicability_rows:
        lines.append(f"- **{item['domain']}**: {item['interpretation']}")
    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            "GADR and ADR differ only in group versus pair weighting. SGADR and SGADR-star add a nonlinear, lambda-indexed transformation; at lambda 1 they are the macro second moment of group ADR and the square of micro ADR. These are aggregation diagnostics, not mechanism-independent reasoning measures.",
            "",
            "The at-least-10 rate is substantive only where a group contains at least ten candidate pairs; the exported structural-possibility field makes this boundary explicit.",
            "",
            "Two-sided columns (`*_with_two_sided_witness`, 2-SAT only) additionally require a solver-verified unsatisfiable clause subset for the negative instance from the separate UNSAT follow-up run (ADR^{+wu}); they use the same denominator and are empty when the model has no follow-up records.",
            "",
            "Witness-aware columns (`ADR_with_witness`, `GADR_with_witness`, `SGADR_with_witness`, `SGADR_star_with_witness`, `group_level_ADR_with_witness_at_least_k_pair_success`) keep the same denominators but credit a pair only when the positive-side prediction also carried a verified constructive witness: a CDCL-checked satisfying assignment for 3-SAT/2-SAT (ADR^{+w}) or an edge-checked proper 3-coloring for graph coloring (ADR^{+a}). The `witness_status_*` columns count verifier outcomes over positive instances predicted positive. `ADR_minus_ADR_with_witness` is the label-without-witness gap.",
            "",
            "## Files",
            "",
            "- `group_metrics_all_lambdas.csv`: all model/domain/slice/lambda rows.",
            "- `group_metrics_primary_lambda1.csv`: primary lambda-1 rows.",
            "- `group_metrics_per_group_lambda1.csv`: per-group audit rows.",
            "- `validation_against_archived.csv`: reproduction checks.",
            "- `applicability.csv`: scenario-level applicability decisions.",
            "- `input_manifest.json`: exact input paths and available fingerprints.",
            "- `analysis_report.json`: machine-readable run summary.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_lambdas(text: str) -> Tuple[float, ...]:
    values = tuple(sorted(set(float(part.strip()) for part in text.split(",") if part.strip())))
    if PRIMARY_LAMBDA not in values:
        raise ValueError(f"Lambda sweep must include primary lambda {PRIMARY_LAMBDA}")
    if any(value < 0 for value in values):
        raise ValueError("Lambda values must be non-negative")
    return values


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=SCRIPT_DIR / "results")
    parser.add_argument(
        "--lambdas",
        default=",".join(str(value) for value in DEFAULT_LAMBDAS),
        help="Comma-separated non-negative lambda values; must include 1.0.",
    )
    parser.add_argument(
        "--skip-input-hashes",
        action="store_true",
        help="Record sizes and paths without content hashes (for a faster development run).",
    )
    args = parser.parse_args()
    lambdas = parse_lambdas(args.lambdas)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    required = [COMMERCIAL_REPORT, TWO_SAT_REPORT, GRAPH_REPORT]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise SystemExit("Missing required archived manifests:\n" + "\n".join(missing))

    commercial, two_sat, graph = load_archive_modules()
    commercial_report = _load_json(COMMERCIAL_REPORT)
    two_sat_report = _load_json(TWO_SAT_REPORT)
    graph_report = _load_json(GRAPH_REPORT)

    rows: List[Dict[str, object]] = []
    group_rows: List[Dict[str, object]] = []
    validation_rows: List[Dict[str, object]] = []
    input_paths_by_domain: Dict[str, List[Path]] = {
        "3-SAT": [COMMERCIAL_REPORT],
        "2-SAT": [TWO_SAT_REPORT],
        "Graph 3-coloring": [GRAPH_REPORT],
    }

    run_three_sat(
        commercial, commercial_report, lambdas, rows, group_rows,
        validation_rows, input_paths_by_domain["3-SAT"],
    )
    run_two_sat(
        commercial, two_sat, two_sat_report, lambdas, rows, group_rows,
        validation_rows, input_paths_by_domain["2-SAT"],
    )
    run_graph_coloring(
        commercial, graph, graph_report, lambdas, rows, group_rows,
        validation_rows, input_paths_by_domain["Graph 3-coloring"],
    )

    primary_rows = [row for row in rows if abs(float(row["sgadr_lambda"]) - PRIMARY_LAMBDA) < 1e-12]
    applicability_rows = build_applicability_rows(primary_rows)

    failed_validations = [
        row for row in validation_rows if not bool(row["matches_within_1e-12"])
    ]
    if failed_validations:
        print(
            f"[audit] {len(failed_validations)} current-input values differ from the archived reports; "
            "exporting the differences instead of replacing current-input results.",
            flush=True,
        )

    domain_specs = {
        "3-SAT": (
            "3sat",
            [COMMERCIAL_ANALYSIS_DIR / "compute_commercial_api_metrics.py"],
        ),
        "2-SAT": (
            "2sat",
            [
                COMMERCIAL_ANALYSIS_DIR / "compute_commercial_api_metrics.py",
                TWO_SAT_ANALYSIS_DIR / "compute_open_source_LLM_2SAT_metrics.py",
            ],
        ),
        "Graph 3-coloring": (
            "graph_coloring",
            [
                COMMERCIAL_ANALYSIS_DIR / "compute_commercial_api_metrics.py",
                GRAPH_ANALYSIS_DIR / "graph_coloring_metrics_common.py",
            ],
        ),
    }
    leading = ["domain", "model", "configuration", "slice_type", "slice_value", "sgadr_lambda"]
    domain_reports = {}
    for domain, (slug, parser_paths) in domain_specs.items():
        domain_dir = output_dir / slug
        domain_dir.mkdir(parents=True, exist_ok=True)
        domain_rows = [row for row in rows if row["domain"] == domain]
        domain_primary_rows = [row for row in primary_rows if row["domain"] == domain]
        domain_group_rows = [row for row in group_rows if row["domain"] == domain]
        domain_validation_rows = [row for row in validation_rows if row["domain"] == domain]
        domain_failed_validations = [
            row for row in domain_validation_rows
            if not bool(row["matches_within_1e-12"])
        ]
        domain_applicability = [row for row in applicability_rows if row["domain"] == domain]

        write_csv(domain_dir / "group_metrics_all_lambdas.csv", domain_rows, leading)
        write_csv(domain_dir / "group_metrics_primary_lambda1.csv", domain_primary_rows, leading)
        write_csv(
            domain_dir / "group_metrics_per_group_lambda1.csv",
            domain_group_rows,
            ["domain", "model", "configuration", "slice_type", "slice_value", "group_id"],
        )
        write_csv(
            domain_dir / "validation_against_archived.csv",
            domain_validation_rows,
            ["domain", "model", "configuration", "slice_type", "slice_value", "metric"],
        )
        write_csv(domain_dir / "applicability.csv", domain_applicability, ["domain"])

        unique_inputs = sorted(
            {path.resolve() for path in input_paths_by_domain[domain]}, key=str
        )
        if args.skip_input_hashes:
            fingerprints = [
                {
                    "path": workspace_relative(path),
                    "kind": (
                        "file" if path.is_file()
                        else "directory" if path.is_dir()
                        else "missing"
                    ),
                }
                for path in unique_inputs
            ]
        else:
            fingerprints = [fingerprint_path(path) for path in unique_inputs]
        input_manifest = {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "workspace_root": str(WORKSPACE_ROOT),
            "domain": domain,
            "inputs": fingerprints,
            "parser_sources": [
                *[fingerprint_path(path) for path in parser_paths],
                fingerprint_path(SCRIPT_DIR / "group_metrics.py"),
                fingerprint_path(Path(__file__).resolve()),
            ],
        }
        (domain_dir / "input_manifest.json").write_text(
            json.dumps(input_manifest, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

        domain_report = {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "command": " ".join(sys.argv),
            "domain": domain,
            "primary_lambda": PRIMARY_LAMBDA,
            "lambdas": list(lambdas),
            "thresholds": list(THRESHOLDS),
            "num_all_lambda_rows": len(domain_rows),
            "num_primary_rows": len(domain_primary_rows),
            "num_per_group_rows": len(domain_group_rows),
            "num_validation_rows": len(domain_validation_rows),
            "all_archived_comparisons_match": not domain_failed_validations,
            "num_archived_comparison_mismatches": len(domain_failed_validations),
            "maximum_archived_absolute_difference": max(
                (
                    float(row["absolute_difference"])
                    for row in domain_failed_validations
                    if row.get("absolute_difference") is not None
                ),
                default=0.0,
            ),
            "applicability": domain_applicability,
        }
        (domain_dir / "analysis_report.json").write_text(
            json.dumps(domain_report, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        write_results_markdown(
            domain_dir / "RESULTS.md",
            domain_rows,
            domain_primary_rows,
            domain_validation_rows,
            domain_applicability,
        )
        domain_reports[slug] = domain_report

    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": " ".join(sys.argv),
        "python": sys.version,
        "platform": platform.platform(),
        "primary_lambda": PRIMARY_LAMBDA,
        "lambdas": list(lambdas),
        "thresholds": list(THRESHOLDS),
        "all_archived_comparisons_match": not failed_validations,
        "num_archived_comparison_mismatches": len(failed_validations),
        "results_are_separated_by_domain": True,
        "domain_output_directories": {
            slug: str(output_dir / slug) for slug, _ in domain_specs.values()
        },
        "domain_reports": domain_reports,
    }
    (output_dir / "analysis_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
