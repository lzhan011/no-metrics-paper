# Group-level analysis

This directory recomputes group-aware label-pair metrics from the prediction
outputs used by the paper. The executed scope is deliberately limited to
3-SAT, 2-SAT, and graph 3-coloring.

## Metric definitions

For group `g`, let `a_g` be the fraction of complete positive/negative pairs
for which both labels are predicted correctly.

- `GADR = mean_g(a_g)` (equal group weighting).
- `ADR` is total successful pairs divided by total complete pairs (equal pair
  weighting).
- `SGADR(lambda) = mean_g[a_g - lambda * a_g * (1-a_g)]`.
- `SGADR_star(lambda) = ADR - lambda * ADR * (1-ADR)`.
- At `lambda=1`, these reduce to `mean_g(a_g^2)` and `ADR^2`.
- `group_level_ADR_at_least_k_pair_success` is the fraction of evaluable groups
  with at least `k` successful pairs. The run exports `k=1,...,10`.

Incomplete pairs remain visible in completion counts but do not enter an ADR
denominator. A successful pair requires both predictions to be present and
correct.

### Witness-aware variants (ADR^{+w} for SAT, ADR^{+a} for graph coloring)

Every metric above has a `*_with_witness` twin with the **same denominator**
but a stricter numerator: a pair counts only if both labels are correct **and**
the positive-side prediction carried a constructive witness that the domain's
established validator verified against the original instance file:

| Domain | Witness | Validator (reused, not modified) |
|---|---|---|
| 3-SAT | satisfying assignment | `compute_commercial_api_metrics.verify_assignment_for_instance` (CDCL via pysat/pycosat) |
| 2-SAT | satisfying assignment | same commercial CDCL checker, as in `compute_open_source_LLM_2SAT_metrics.build_open_source_pairs` |
| Graph 3-coloring | proper 3-coloring | `graph_coloring_metrics_common.verify_assignment_for_instance` (edge check) |

Exported columns: `ADR_with_witness`, `GADR_with_witness`, `SGADR_with_witness`,
`SGADR_star_with_witness`, `pair_success_with_witness_total`,
`group_level_ADR_with_witness_at_least_k_pair_success` (k=1..10),
`ADR_minus_ADR_with_witness`, `GADR_minus_GADR_with_witness`, and per-group
`pair_success_with_witness_count`, `ADR_with_witness_g`, `SGADR_with_witness_g`.
The `witness_status_*` columns count validator outcomes over positive
instances predicted positive (`satisfied`, `unsatisfied`, `missing_assignment`,
`invalid_assignment`, `missing_cnf`, `cdcl_unavailable`), so the witness
denominator is auditable. The validator is invoked exactly as the archived
pair builders do: only for a positive instance predicted positive.

**Two-sided variant for 2-SAT (ADR^{+wu}).** `*_with_two_sided_witness`
additionally requires, for the negative (UNSAT) instance, that the separate
UNSAT clause-subset follow-up run (`..._unsat_clause_subset_followup_...`,
path taken from the archived report's `unsat_clause_subset_followup_results`)
returned a solver-verified unsatisfiable clause subset. Records are matched by
dataset-relative CNF path exactly as `build_open_source_pairs` does. The
denominator is still all complete pairs (as in the archived implementation),
so ADR^{+wu} <= ADR^{+w}. When a model has no follow-up records the two-sided
columns are empty (None), never 0; `num_pairs_negative_witness_available` /
`_verified` give the coverage. Paper Table `tab:group-adr-overall` uses the
one-sided ADR^{+w} as W for 2-SAT; the two-sided columns are exported in the
CSVs for reference. `summarize_w_gap.py` prints every statistic quoted in the
accompanying paragraph (`--two-sided` switches 2-SAT to ADR^{+wu}).

`validation_against_archived.csv` compares each `*_with_witness` value with the
archived `*_with_assignment` value from the source report (column
`archived_metric`). `check_witness_results.py` re-checks after a run that every
witness metric is bounded by its label-only twin and prints the comparison.

## Code and inputs

- `group_metrics.py` contains the domain-neutral implementation.
- `run_group_analysis.py` maps the paper model configurations to their raw
  prediction outputs and reuses the established parsers/deduplication code.
- `test_group_metrics.py` contains hand-computed tests for incompleteness,
  duplicate rejection, and the lambda identities.
- Each result subdirectory has its own `input_manifest.json`, which lists the
  exact archived manifest, prediction files/directories, and parser sources for
  that domain.

No model inference is performed. The inputs are existing prediction records.

## Run

```bash
cd No_Metrics/analysis/group_analysis   # from the repository root
python -m unittest -v test_group_metrics.py
set -o pipefail
python run_group_analysis.py --output-dir results --skip-input-hashes 2>&1 | tee run.log
python check_witness_results.py
python summarize_w_gap.py
```

Omit `--skip-input-hashes` when full content hashing of every prediction input
is required.

## Separate outputs

The numerical results are intentionally not combined across domains:

- `results/3sat/`
- `results/2sat/`
- `results/graph_coloring/`

Each directory contains all-lambda and primary-lambda CSVs, per-group audit
rows, archived-report comparisons, an applicability record, an input manifest,
a JSON run report, and `RESULTS.md`.

The current 3-SAT and graph-coloring outputs exactly reproduce their archived
reports. Some current 2-SAT prediction records differ from values embedded in
the older archived report. The recomputation matches the repository's original
2-SAT aggregation code on the current records; all old-versus-current
differences are retained in `results/2sat/validation_against_archived.csv`.
