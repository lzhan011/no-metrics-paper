#!/usr/bin/env python3
"""Group-level aggregation metrics for matched positive/negative pairs.

This module is intentionally domain-neutral. Domain adapters only need to
produce :class:`PairOutcome` records after applying their established parsing
and deduplication rules.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from math import fsum
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


@dataclass(frozen=True)
class PairOutcome:
    """One intended positive/negative pair within a named group."""

    group_id: str
    pair_id: str
    positive_prediction_available: bool
    negative_prediction_available: bool
    positive_correct: bool
    negative_correct: bool
    # Witness-aware flag: True only when the positive-side prediction carried a
    # constructive artifact (SAT assignment / graph coloring) that the domain
    # validator verified against the original instance.  Label-only metrics
    # ignore this field; ``*_with_witness`` metrics require it.
    positive_witness_verified: bool = False
    # Negative-side witness (2-SAT follow-up): True/False when a follow-up
    # record exists for the negative instance and the returned unsatisfiable
    # clause subset was / was not solver-verified; None when no follow-up
    # record is available for this pair.
    negative_witness_verified: Optional[bool] = None

    @property
    def complete(self) -> bool:
        return self.positive_prediction_available and self.negative_prediction_available

    @property
    def success(self) -> bool:
        return self.complete and self.positive_correct and self.negative_correct

    @property
    def success_with_witness(self) -> bool:
        """Label-pair success plus a verified positive-side witness (ADR^{+w})."""
        return self.success and self.positive_witness_verified

    @property
    def negative_witness_available(self) -> bool:
        return self.negative_witness_verified is not None

    @property
    def success_with_two_sided_witness(self) -> bool:
        """Label-pair success plus verified witnesses on both sides (ADR^{+wu})."""
        return self.success_with_witness and bool(self.negative_witness_verified)


def _mean(values: Sequence[float]) -> Optional[float]:
    if not values:
        return None
    return fsum(values) / len(values)


def _safe_div(numerator: float, denominator: float) -> Optional[float]:
    if denominator == 0:
        return None
    return numerator / denominator


def compute_group_metrics(
    pairs: Iterable[PairOutcome],
    sgadr_lambda: float = 1.0,
    thresholds: Iterable[int] = range(1, 11),
) -> Tuple[Dict[str, object], List[Dict[str, object]]]:
    """Compute group-equal, pair-equal, nonlinear, and thresholded metrics.

    Let ``a_g`` be successful complete pairs divided by complete pairs in
    evaluable group ``g``. The returned metrics follow the archived code:

    ``GADR = mean_g(a_g)``

    ``ADR = sum_g(success_g) / sum_g(complete_g)``

    ``SGADR(lambda) = mean_g[a_g - lambda * a_g * (1-a_g)]``

    ``SGADR_star(lambda) = ADR - lambda * ADR * (1-ADR)``

    At lambda 1, the last two reduce to ``mean_g(a_g^2)`` and ``ADR^2``.
    Threshold metrics use the same denominator as GADR: groups with at least
    one complete pair.
    """

    if sgadr_lambda < 0:
        raise ValueError("sgadr_lambda must be non-negative")

    threshold_values = sorted(set(int(k) for k in thresholds))
    if not threshold_values or threshold_values[0] < 1:
        raise ValueError("thresholds must contain positive integers")

    grouped: Dict[str, List[PairOutcome]] = defaultdict(list)
    seen_keys = set()
    for pair in pairs:
        key = (str(pair.group_id), str(pair.pair_id))
        if key in seen_keys:
            raise ValueError(f"duplicate pair key after adapter deduplication: {key}")
        seen_keys.add(key)
        grouped[str(pair.group_id)].append(pair)

    group_rows: List[Dict[str, object]] = []
    for group_id in sorted(grouped):
        group_pairs = grouped[group_id]
        complete_pairs = [pair for pair in group_pairs if pair.complete]
        success_count = sum(1 for pair in complete_pairs if pair.success)
        witness_success_count = sum(
            1 for pair in complete_pairs if pair.success_with_witness
        )
        two_sided_success_count = sum(
            1 for pair in complete_pairs if pair.success_with_two_sided_witness
        )
        negative_available_count = sum(
            1 for pair in complete_pairs if pair.negative_witness_available
        )
        negative_verified_count = sum(
            1 for pair in complete_pairs if bool(pair.negative_witness_verified)
        )
        complete_count = len(complete_pairs)
        adr_wu_g = _safe_div(two_sided_success_count, complete_count)
        sgadr_wu_g = (
            adr_wu_g - sgadr_lambda * adr_wu_g * (1.0 - adr_wu_g)
            if adr_wu_g is not None
            else None
        )
        adr_g = _safe_div(success_count, complete_count)
        adr_w_g = _safe_div(witness_success_count, complete_count)
        sgadr_w_g = (
            adr_w_g - sgadr_lambda * adr_w_g * (1.0 - adr_w_g)
            if adr_w_g is not None
            else None
        )
        bernoulli_variance_g = (
            adr_g * (1.0 - adr_g) if adr_g is not None else None
        )
        sgadr_g = (
            adr_g - sgadr_lambda * bernoulli_variance_g
            if adr_g is not None and bernoulli_variance_g is not None
            else None
        )
        row: Dict[str, object] = {
            "group_id": group_id,
            "k_g_total_pairs": len(group_pairs),
            "k_g_complete_pairs": complete_count,
            "pair_success_count": success_count,
            "ADR_g": adr_g,
            "ADR_bernoulli_variance_g": bernoulli_variance_g,
            "SGADR_g": sgadr_g,
            "sgadr_lambda": sgadr_lambda,
            "pair_success_with_witness_count": witness_success_count,
            "ADR_with_witness_g": adr_w_g,
            "SGADR_with_witness_g": sgadr_w_g,
            "pair_negative_witness_available_count": negative_available_count,
            "pair_negative_witness_verified_count": negative_verified_count,
            "pair_success_with_two_sided_witness_count": two_sided_success_count,
            "ADR_with_two_sided_witness_g": adr_wu_g,
            "SGADR_with_two_sided_witness_g": sgadr_wu_g,
        }
        for threshold in threshold_values:
            row[f"passes_at_least_{threshold}"] = (
                success_count >= threshold if adr_g is not None else None
            )
            row[f"passes_with_witness_at_least_{threshold}"] = (
                witness_success_count >= threshold if adr_g is not None else None
            )
            row[f"passes_with_two_sided_witness_at_least_{threshold}"] = (
                two_sided_success_count >= threshold if adr_g is not None else None
            )
        group_rows.append(row)

    valid_rows = [row for row in group_rows if row["ADR_g"] is not None]
    group_adrs = [float(row["ADR_g"]) for row in valid_rows]
    group_sgadrs = [float(row["SGADR_g"]) for row in valid_rows]

    total_pairs = sum(int(row["k_g_total_pairs"]) for row in group_rows)
    total_complete = sum(int(row["k_g_complete_pairs"]) for row in group_rows)
    total_success = sum(int(row["pair_success_count"]) for row in group_rows)
    total_witness_success = sum(
        int(row["pair_success_with_witness_count"]) for row in group_rows
    )
    group_adrs_w = [float(row["ADR_with_witness_g"]) for row in valid_rows]
    group_sgadrs_w = [float(row["SGADR_with_witness_g"]) for row in valid_rows]
    gadr_w = _mean(group_adrs_w)
    adr_w = _safe_div(total_witness_success, total_complete)
    sgadr_w = _mean(group_sgadrs_w)
    sgadr_star_w = (
        adr_w - sgadr_lambda * adr_w * (1.0 - adr_w) if adr_w is not None else None
    )
    total_negative_available = sum(
        int(row["pair_negative_witness_available_count"]) for row in group_rows
    )
    total_negative_verified = sum(
        int(row["pair_negative_witness_verified_count"]) for row in group_rows
    )
    total_two_sided_success = sum(
        int(row["pair_success_with_two_sided_witness_count"]) for row in group_rows
    )
    # Two-sided quantities are reported only when at least one complete pair
    # in the slice has a negative-side follow-up record; otherwise they are
    # None ("--"), never 0, so absence of the follow-up is not read as failure.
    two_sided_defined = total_negative_available > 0
    gadr_wu = (
        _mean([float(row["ADR_with_two_sided_witness_g"]) for row in valid_rows])
        if two_sided_defined
        else None
    )
    adr_wu = _safe_div(total_two_sided_success, total_complete) if two_sided_defined else None
    sgadr_wu = (
        _mean([float(row["SGADR_with_two_sided_witness_g"]) for row in valid_rows])
        if two_sided_defined
        else None
    )
    sgadr_star_wu = (
        adr_wu - sgadr_lambda * adr_wu * (1.0 - adr_wu) if adr_wu is not None else None
    )

    gadr = _mean(group_adrs)
    adr = _safe_div(total_success, total_complete)
    mean_bernoulli_variance = _mean(
        [float(row["ADR_bernoulli_variance_g"]) for row in valid_rows]
    )
    sgadr = _mean(group_sgadrs)
    sgadr_star = (
        adr - sgadr_lambda * adr * (1.0 - adr) if adr is not None else None
    )
    macro_second_moment = _mean([value * value for value in group_adrs])
    micro_square = adr * adr if adr is not None else None
    between_group_variance = (
        _mean([(value - gadr) ** 2 for value in group_adrs])
        if gadr is not None
        else None
    )

    metrics: Dict[str, object] = {
        "sgadr_lambda": sgadr_lambda,
        "num_groups_total": len(group_rows),
        "num_groups_with_complete_pairs": len(valid_rows),
        "num_pairs_total": total_pairs,
        "num_complete_pairs_total": total_complete,
        "pair_success_both_correct_total": total_success,
        "group_coverage": _safe_div(len(valid_rows), len(group_rows)),
        "pair_completion_rate": _safe_div(total_complete, total_pairs),
        "GADR": gadr,
        "ADR": adr,
        "SGADR": sgadr,
        "SGADR_star": sgadr_star,
        "mean_group_bernoulli_variance": mean_bernoulli_variance,
        "between_group_ADR_variance": between_group_variance,
        "macro_second_moment_ADR_g": macro_second_moment,
        "micro_ADR_square": micro_square,
        "SGADR_minus_SGADR_star": (
            sgadr - sgadr_star
            if sgadr is not None and sgadr_star is not None
            else None
        ),
        "GADR_minus_ADR": (
            gadr - adr if gadr is not None and adr is not None else None
        ),
        "max_pairs_in_group": max(
            (int(row["k_g_total_pairs"]) for row in group_rows), default=0
        ),
        "max_complete_pairs_in_group": max(
            (int(row["k_g_complete_pairs"]) for row in group_rows), default=0
        ),
        # Witness-aware family (ADR^{+w} / ADR^{+a}): same denominators as the
        # label-only family, numerator additionally requires a verified
        # positive-side witness.
        "pair_success_with_witness_total": total_witness_success,
        "GADR_with_witness": gadr_w,
        "ADR_with_witness": adr_w,
        "SGADR_with_witness": sgadr_w,
        "SGADR_star_with_witness": sgadr_star_w,
        "ADR_minus_ADR_with_witness": (
            adr - adr_w if adr is not None and adr_w is not None else None
        ),
        "GADR_minus_GADR_with_witness": (
            gadr - gadr_w if gadr is not None and gadr_w is not None else None
        ),
        # Two-sided witness family (ADR^{+wu}): additionally requires a
        # solver-verified unsatisfiable clause subset for the negative
        # instance, obtained from a separate follow-up run. Same denominator
        # as the label-only family (all complete pairs), as in the archived
        # 2-SAT implementation; None when no follow-up record exists.
        "num_pairs_negative_witness_available": total_negative_available,
        "num_pairs_negative_witness_verified": total_negative_verified,
        "pair_success_with_two_sided_witness_total": (
            total_two_sided_success if two_sided_defined else None
        ),
        "GADR_with_two_sided_witness": gadr_wu,
        "ADR_with_two_sided_witness": adr_wu,
        "SGADR_with_two_sided_witness": sgadr_wu,
        "SGADR_star_with_two_sided_witness": sgadr_star_wu,
    }

    for threshold in threshold_values:
        metric_name = f"group_level_ADR_at_least_{threshold}_pair_success"
        metrics[metric_name] = (
            _safe_div(
                sum(
                    1
                    for row in valid_rows
                    if int(row["pair_success_count"]) >= threshold
                ),
                len(valid_rows),
            )
            if valid_rows
            else None
        )
        metrics[f"group_level_ADR_with_witness_at_least_{threshold}_pair_success"] = (
            _safe_div(
                sum(
                    1
                    for row in valid_rows
                    if int(row["pair_success_with_witness_count"]) >= threshold
                ),
                len(valid_rows),
            )
            if valid_rows
            else None
        )
        metrics[f"group_level_ADR_with_two_sided_witness_at_least_{threshold}_pair_success"] = (
            _safe_div(
                sum(
                    1
                    for row in valid_rows
                    if int(row["pair_success_with_two_sided_witness_count"]) >= threshold
                ),
                len(valid_rows),
            )
            if valid_rows and two_sided_defined
            else None
        )
        metrics[f"num_groups_with_at_least_{threshold}_candidate_pairs"] = sum(
            1 for row in valid_rows if int(row["k_g_total_pairs"]) >= threshold
        )
        metrics[f"num_groups_with_at_least_{threshold}_complete_pairs"] = sum(
            1 for row in valid_rows if int(row["k_g_complete_pairs"]) >= threshold
        )

    return metrics, group_rows

