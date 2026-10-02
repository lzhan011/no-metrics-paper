#!/usr/bin/env bash
# Recreate the evaluation trees from data/*.tar.gz (idempotent).
set -euo pipefail
cd "$(dirname "$0")/.."
for f in data/*.tar.gz; do
  echo "unpacking $f"
  tar -xzf "$f" -C .
done
echo "done: $(find SAT_Group_Evaluation/commercial_api SAT_Group_Evaluation/open_source_LLM_2SAT SAT_Group_Evaluation/generate Graph_Colouring_Evaluation -type f 2>/dev/null | wc -l) files"
