# Correct Predictions, Wrong Reasons — paper, code and data

Source, data and reproduction scripts for the paper
*Correct Predictions, Wrong Reasons: On the Limits of Metrics for Measuring
Reasoning Ability* (`No_Metrics/paper/no-metrics-main-FSE2027.tex`, with a
supplementary document next to it).

The repository is laid out like the original evaluation workspace so that the
archived analysis code runs unchanged:

```
No_Metrics/
  paper/                      LaTeX sources, bibliography, figures, figure scripts
  analysis/group_analysis/    group-level and witness-aware metric recomputation (Table 6 etc.)
SAT_Group_Evaluation/
  analysis/                   frozen parsers + archived reports/CSVs for 3-SAT and 2-SAT
  RMC_Paper_ICSE2027_supplementary.tex, CSA_Paper_ICSE2027_supplementary.tex
                              frozen archive tables that the paper reanalyses (Figure 1, RQ1–RQ3)
  commercial_api/, open_source_LLM_2SAT/, generate/     <- unpacked from data/ (git-ignored)
Graph_Colouring_Evaluation/   <- unpacked from data/ (git-ignored)
data/                         gzip tarballs with the raw prediction records and the
                              CNF instances they reference (each < 100 MB)
scripts/                      unpack_data.sh, run_all.sh, build_data_archives.py
```

## Quick start

```bash
pip install -r requirements.txt        # python 3.10+, pycosat (or python-sat), matplotlib
bash scripts/unpack_data.sh            # recreates the evaluation trees from data/*.tar.gz
bash scripts/run_all.sh                # tests -> group analysis (~10 min) -> checks -> figures -> PDF
```

`run_all.sh` needs a TeX distribution with `latexmk`, `acmart` dependencies and
`minted` (Pygments) for the last step; everything before it is pure Python.

## What is reproduced

| Paper element | Produced by | From |
|---|---|---|
| Table 6 (group-level metrics incl. witness columns `W`, `A^W`) | `No_Metrics/analysis/group_analysis/run_group_analysis.py` → `results/*/group_metrics_primary_lambda1.csv` | raw prediction records in `data/`, archived reports in `SAT_Group_Evaluation/analysis/` |
| Table 5 last column (`ADR^{+wu}_all`) | `table5_wu_whole_set.py` | same results |
| Paragraph "Group metrics versus witness evidence" | `summarize_w_gap.py` | same results |
| Figure 1 (`fig_*_caseb_byN`) | `No_Metrics/paper/plot_shortcut_margin_from_latex.py` | tables `tab:com_pern_caseB` / `tab:os_pern_caseB` in `SAT_Group_Evaluation/RMC_Paper_ICSE2027_supplementary.tex` |
| Figure 2 (`fig_*_casec_byN`) | `No_Metrics/paper/plot_shortcut_residualization_sensitivity.py` | `rmc_filtered_*` / `rmc_paper_open_vendor_case_*` CSVs |
| Tables 1–5, 7–8 and RQ1–RQ3 numbers | copied from the frozen archives | `CSA_Paper_ICSE2027_supplementary.tex`, `RMC_Paper_ICSE2027_supplementary.tex`, archived CSVs |

The committed `results/` directory is the output of the last full run; re-running
regenerates it (the committed figures are pixel-identical to a fresh run).

## Data

`data/*.tar.gz` contain, with paths relative to the repository root:

* `predictions_3sat_commercial` — GPT-5, Claude Opus 4.7 (medium thinking) and
  DeepSeek-Reasoner per-instance outputs on 3-SAT (`by_file` trees).
* `predictions_2sat_<model>` — `results_desc.jsonl` of the six open-weight 2-SAT
  runs plus, where it exists, the UNSAT clause-subset follow-up run.
* `predictions_gc_<model>` — `results_desc.jsonl` of the eight graph-colouring runs.
* `dataset_{3sat,2sat,gc}_subset` — only the CNF instances (and sidecar
  metadata) referenced by those prediction records; the witness validators
  read them. The full generators and datasets live in the RMC/CSA repositories.

`scripts/build_data_archives.py` is the maintainer tool that produced these
tarballs from the original workspace; `scripts/needed_cnf.json` lists the
referenced instances.

## Provenance notes

* The archived report JSONs under `SAT_Group_Evaluation/analysis/` and the two
  archive `.tex` files are kept verbatim; they contain absolute paths of the
  machine they were produced on. `run_group_analysis.py` re-roots every such
  path at the repository root (`rebase_path`), so nothing has to be edited.
* `validation_against_archived.csv` in each results directory records the
  comparison of every recomputed value with the archived reports.
* The analysis performs no model inference; all inputs are frozen records.
