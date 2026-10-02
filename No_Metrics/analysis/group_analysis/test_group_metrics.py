#!/usr/bin/env python3
"""Focused tests for the domain-neutral group aggregation formulas."""

import math
import unittest

from group_metrics import PairOutcome, compute_group_metrics


def pair(
    group: str,
    pair_id: str,
    success: bool,
    complete: bool = True,
    witness: bool = False,
    neg_witness=None,
) -> PairOutcome:
    return PairOutcome(
        group_id=group,
        pair_id=pair_id,
        positive_prediction_available=complete,
        negative_prediction_available=complete,
        positive_correct=success,
        negative_correct=success,
        positive_witness_verified=witness,
        negative_witness_verified=neg_witness,
    )


class GroupMetricTests(unittest.TestCase):
    def test_lambda_one_matches_square_definitions(self):
        pairs = [
            pair("a", "1", True),
            pair("a", "2", True),
            pair("b", "1", False),
        ]
        metrics, groups = compute_group_metrics(pairs, sgadr_lambda=1.0)
        self.assertEqual(len(groups), 2)
        self.assertAlmostEqual(metrics["GADR"], 0.5)
        self.assertAlmostEqual(metrics["ADR"], 2.0 / 3.0)
        self.assertAlmostEqual(metrics["SGADR"], 0.5)
        self.assertAlmostEqual(metrics["SGADR_star"], (2.0 / 3.0) ** 2)
        self.assertAlmostEqual(metrics["between_group_ADR_variance"], 0.25)
        self.assertAlmostEqual(metrics["group_level_ADR_at_least_1_pair_success"], 0.5)
        self.assertAlmostEqual(metrics["group_level_ADR_at_least_2_pair_success"], 0.5)
        self.assertEqual(metrics["group_level_ADR_at_least_10_pair_success"], 0.0)

    def test_incomplete_pairs_are_visible_but_not_denominator_credit(self):
        pairs = [pair("a", "1", True), pair("a", "2", False, complete=False)]
        metrics, groups = compute_group_metrics(pairs, sgadr_lambda=1.0)
        self.assertEqual(metrics["num_pairs_total"], 2)
        self.assertEqual(metrics["num_complete_pairs_total"], 1)
        self.assertEqual(metrics["pair_success_both_correct_total"], 1)
        self.assertEqual(metrics["ADR"], 1.0)
        self.assertEqual(groups[0]["k_g_total_pairs"], 2)
        self.assertEqual(groups[0]["k_g_complete_pairs"], 1)

    def test_lambda_family_identity(self):
        pairs = [pair("a", "1", True), pair("a", "2", False)]
        metrics, _ = compute_group_metrics(pairs, sgadr_lambda=0.5)
        expected = 0.5 - 0.5 * 0.5 * 0.5
        self.assertTrue(math.isclose(metrics["SGADR"], expected))
        self.assertTrue(math.isclose(metrics["SGADR_star"], expected))

    def test_witness_metrics_require_label_success_and_verified_witness(self):
        pairs = [
            pair("a", "1", True, witness=True),
            pair("a", "2", True, witness=False),
            pair("b", "1", False, witness=True),   # witness flag without label success: no credit
            pair("b", "2", True, witness=True),
        ]
        metrics, groups = compute_group_metrics(pairs, sgadr_lambda=1.0)
        # label-only: a=2/2, b=1/2
        self.assertAlmostEqual(metrics["ADR"], 3.0 / 4.0)
        self.assertAlmostEqual(metrics["GADR"], (1.0 + 0.5) / 2.0)
        # witness-aware: a=1/2, b=1/2
        self.assertEqual(metrics["pair_success_with_witness_total"], 2)
        self.assertAlmostEqual(metrics["ADR_with_witness"], 2.0 / 4.0)
        self.assertAlmostEqual(metrics["GADR_with_witness"], 0.5)
        self.assertAlmostEqual(metrics["SGADR_with_witness"], 0.25)
        self.assertAlmostEqual(metrics["SGADR_star_with_witness"], 0.25)
        self.assertAlmostEqual(metrics["ADR_minus_ADR_with_witness"], 0.25)
        self.assertAlmostEqual(
            metrics["group_level_ADR_with_witness_at_least_1_pair_success"], 1.0
        )
        self.assertAlmostEqual(
            metrics["group_level_ADR_with_witness_at_least_2_pair_success"], 0.0
        )
        self.assertEqual(groups[0]["pair_success_with_witness_count"], 1)
        self.assertLessEqual(metrics["ADR_with_witness"], metrics["ADR"])

    def test_witness_default_false_keeps_label_metrics_unchanged(self):
        pairs = [pair("a", "1", True), pair("a", "2", False)]
        metrics, _ = compute_group_metrics(pairs, sgadr_lambda=1.0)
        self.assertAlmostEqual(metrics["ADR"], 0.5)
        self.assertEqual(metrics["ADR_with_witness"], 0.0)
        self.assertEqual(metrics["pair_success_with_witness_total"], 0)

    def test_two_sided_witness_requires_both_sides_and_uses_full_denominator(self):
        pairs = [
            pair("a", "1", True, witness=True, neg_witness=True),   # counts
            pair("a", "2", True, witness=True, neg_witness=False),  # UNSAT core not verified
            pair("a", "3", True, witness=False, neg_witness=True),  # no SAT witness
            pair("b", "1", True, witness=True, neg_witness=None),   # no follow-up record
        ]
        metrics, groups = compute_group_metrics(pairs, sgadr_lambda=1.0)
        self.assertEqual(metrics["num_pairs_negative_witness_available"], 3)
        self.assertEqual(metrics["num_pairs_negative_witness_verified"], 2)
        self.assertEqual(metrics["pair_success_with_two_sided_witness_total"], 1)
        self.assertAlmostEqual(metrics["ADR_with_two_sided_witness"], 1.0 / 4.0)
        self.assertAlmostEqual(metrics["GADR_with_two_sided_witness"], (1.0 / 3.0 + 0.0) / 2.0)
        self.assertLessEqual(metrics["ADR_with_two_sided_witness"], metrics["ADR_with_witness"])
        self.assertAlmostEqual(
            metrics["group_level_ADR_with_two_sided_witness_at_least_1_pair_success"], 0.5
        )
        self.assertEqual(groups[0]["pair_success_with_two_sided_witness_count"], 1)

    def test_two_sided_witness_is_none_without_any_followup(self):
        pairs = [pair("a", "1", True, witness=True), pair("a", "2", False)]
        metrics, _ = compute_group_metrics(pairs, sgadr_lambda=1.0)
        self.assertIsNone(metrics["ADR_with_two_sided_witness"])
        self.assertIsNone(metrics["pair_success_with_two_sided_witness_total"])
        self.assertIsNone(metrics["group_level_ADR_with_two_sided_witness_at_least_1_pair_success"])
        self.assertEqual(metrics["num_pairs_negative_witness_available"], 0)

    def test_duplicate_pair_keys_rejected(self):
        with self.assertRaises(ValueError):
            compute_group_metrics([pair("a", "1", True), pair("a", "1", False)])


if __name__ == "__main__":
    unittest.main()

