# Group-analysis results

## Run scope

- Lambda settings: 0.0, 0.25, 0.5, 0.75, 1.0
- Primary lambda: 1.0
- Primary metric slices: 72
- Archived-value comparisons: 1603
- Archived-value mismatches: 307
- Maximum absolute reproduction difference: 237

## Domain coverage

| Domain | Primary slices | Models/configurations | Threshold-10 structurally possible |
|---|---:|---:|---:|
| 2-SAT | 72 | 6 | 72 |

## Applicability

- **2-SAT**: Natural generator group; up to 11 matched SAT/UNSAT pairs.

## Interpretation boundary

GADR and ADR differ only in group versus pair weighting. SGADR and SGADR-star add a nonlinear, lambda-indexed transformation; at lambda 1 they are the macro second moment of group ADR and the square of micro ADR. These are aggregation diagnostics, not mechanism-independent reasoning measures.

The at-least-10 rate is substantive only where a group contains at least ten candidate pairs; the exported structural-possibility field makes this boundary explicit.

Two-sided columns (`*_with_two_sided_witness`, 2-SAT only) additionally require a solver-verified unsatisfiable clause subset for the negative instance from the separate UNSAT follow-up run (ADR^{+wu}); they use the same denominator and are empty when the model has no follow-up records.

Witness-aware columns (`ADR_with_witness`, `GADR_with_witness`, `SGADR_with_witness`, `SGADR_star_with_witness`, `group_level_ADR_with_witness_at_least_k_pair_success`) keep the same denominators but credit a pair only when the positive-side prediction also carried a verified constructive witness: a CDCL-checked satisfying assignment for 3-SAT/2-SAT (ADR^{+w}) or an edge-checked proper 3-coloring for graph coloring (ADR^{+a}). The `witness_status_*` columns count verifier outcomes over positive instances predicted positive. `ADR_minus_ADR_with_witness` is the label-without-witness gap.

## Files

- `group_metrics_all_lambdas.csv`: all model/domain/slice/lambda rows.
- `group_metrics_primary_lambda1.csv`: primary lambda-1 rows.
- `group_metrics_per_group_lambda1.csv`: per-group audit rows.
- `validation_against_archived.csv`: reproduction checks.
- `applicability.csv`: scenario-level applicability decisions.
- `input_manifest.json`: exact input paths and available fingerprints.
- `analysis_report.json`: machine-readable run summary.
