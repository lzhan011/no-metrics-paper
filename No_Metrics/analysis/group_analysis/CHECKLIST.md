# Group-analysis checklist

- [x] Locate the original group metric implementation.
- [x] Lock the definitions of GADR, ADR, SGADR, SGADR-star, and A>=k.
- [x] Resolve the exact paper model inputs from archived manifests.
- [x] Confirm group and pair identifiers for 3-SAT, 2-SAT, and graph coloring.
- [x] Implement neutral reusable group metric functions.
- [x] Implement domain adapters that reuse archived parsing/deduplication.
- [x] Add unit tests for hand-computed examples and lambda identities.
- [x] Narrow execution to 3-SAT, 2-SAT, and graph coloring as requested.
- [x] Configure separate output directories for all three domains.
- [x] Run focused unit and syntax tests after scope changes.
- [x] Run the full selected paper-model analysis.
- [x] Compare lambda-1 results with archived outputs.
- [x] Diagnose the 2-SAT archived/current-input discrepancy against the
  repository's original aggregation implementation.
- [x] Export each domain's per-slice, per-group, applicability, manifest, and
  report files separately.
- [x] Record structural threshold applicability for every slice.
- [x] Write a concise result and reproduction guide.
- [x] (2026-09-29) Add two-sided witness variants (`*_with_two_sided_witness`,
  ADR^{+wu}) for 2-SAT from the archived UNSAT clause-subset follow-up files;
  Llama-Nemotron and Nemotron-H have no follow-up run (values undefined,
  exported empty).
- [x] (2026-09-29) `summarize_w_gap.py`: statistics for the paper paragraph
  "Group metrics versus witness evidence" (W = ADR^{+w} / ADR^{+wu} / ADR^{+a}).
- [x] (2026-09-29) Paper Table 6 uses the one-sided ADR^{+w} as W for 2-SAT;
  two-sided columns remain exported.
