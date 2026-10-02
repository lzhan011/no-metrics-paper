#!/usr/bin/env python3
"""ADR^{+wu} with the whole pair set as denominator, for the rows of paper
Table `tab:two-sided-witness` (Table 5).

For each (model, N) row: numerator = complete pairs whose SAT side has a
CDCL-verified assignment AND whose UNSAT side has a solver-verified
unsatisfiable clause subset from the follow-up run; denominator = ALL complete
pairs at that N (pairs without any follow-up record count as failures).
Values come from results/2sat/group_metrics_primary_lambda1.csv produced by
run_group_analysis.py on the current prediction files.
Prints one LaTeX-ready value per row plus the underlying counts.
"""
import csv
from pathlib import Path

ROWS = [  # (paper alias, model key in results, N)
    ("gpt-oss-20b (reason)", "gpt-oss-20b (reason)", "3"),
    ("Qwen3-14B (think)", "Qwen3-14B (think)", "3"),
    ("Qwen3-32B (think)", "Qwen3-32B (think)", "10"),
    ("gpt-oss-20b (reason)", "gpt-oss-20b (reason)", "10"),
    ("gpt-oss-20b (reason)", "gpt-oss-20b (reason)", "15"),
    ("QwQ-32B (think)", "QwQ-32B (think)", "20"),
]

def main():
    path = Path(__file__).resolve().parent / "results" / "2sat" / "group_metrics_primary_lambda1.csv"
    rows = {(r["model"], r["slice_type"], r["slice_value"]): r for r in csv.DictReader(open(path))}
    print(f"{'model':<22}{'N':>3} {'complete':>9} {'followup':>9} {'succ_wu':>8} {'ADR+wu_all':>11}")
    for alias, key, n in ROWS:
        r = rows[(key, "N", n)]
        complete = int(r["num_complete_pairs_total"])
        avail = int(r["num_pairs_negative_witness_available"])
        succ = r["pair_success_with_two_sided_witness_total"]
        succ = int(float(succ)) if succ not in ("", None) else 0
        value = succ / complete if complete else float("nan")
        print(f"{alias:<22}{n:>3} {complete:>9} {avail:>9} {succ:>8} {value:>11.3f}")

if __name__ == "__main__":
    main()
