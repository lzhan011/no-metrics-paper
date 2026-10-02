#!/usr/bin/env bash
# Full reproduction: unit tests -> group analysis -> checks -> Figures 1/2 -> paper PDF.
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT=$(pwd)
PY=${PYTHON:-python3}
echo "== 1/5 unit tests"
( cd No_Metrics/analysis/group_analysis && $PY -m unittest -v test_group_metrics.py )
echo "== 2/5 group analysis (3-SAT, 2-SAT, graph coloring; ~10 min)"
( cd No_Metrics/analysis/group_analysis && $PY run_group_analysis.py --output-dir results --skip-input-hashes 2>&1 | tee run.log )
echo "== 3/5 result checks"
( cd No_Metrics/analysis/group_analysis && $PY check_witness_results.py && $PY summarize_w_gap.py && $PY table5_wu_whole_set.py )
echo "== 4/5 figures"
( cd No_Metrics/paper && $PY plot_shortcut_margin_from_latex.py ../../SAT_Group_Evaluation/RMC_Paper_ICSE2027_supplementary.tex figures && mv figures/fig_*sat_caseb_byN.* figures/shortcut_margin/ && $PY plot_shortcut_residualization_sensitivity.py )
echo "== 5/5 paper"
( cd No_Metrics/paper && latexmk -pdf -shell-escape -interaction=nonstopmode no-metrics-main-FSE2027.tex && latexmk -pdf -shell-escape -interaction=nonstopmode no-metrics-supplementary-FSE2027.tex )
echo "ALL DONE"
