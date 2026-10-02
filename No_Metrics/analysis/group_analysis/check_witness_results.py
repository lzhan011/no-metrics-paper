#!/usr/bin/env python3
"""Post-run check: witness metrics present, bounded by label metrics, and
compared with the archived *_with_assignment values."""
import csv, sys
from pathlib import Path
root = Path(__file__).resolve().parent / "results"
bad = 0
for slug in ("3sat", "2sat", "graph_coloring"):
    rows = list(csv.DictReader(open(root / slug / "group_metrics_primary_lambda1.csv")))
    ov = [r for r in rows if r["slice_type"] == "overall"]
    print(f"## {slug}: {len(rows)} primary rows, {len(ov)} overall rows")
    for r in ov:
        adr, adrw = float(r["ADR"]), float(r["ADR_with_witness"])
        gadr, gadrw = float(r["GADR"]), float(r["GADR_with_witness"])
        if adrw > adr + 1e-12 or gadrw > gadr + 1e-12:
            bad += 1; flag = "  <-- VIOLATION"
        else:
            flag = ""
        st = {k[len("witness_status_"):]: r[k] for k in r if k.startswith("witness_status_")}
        print(f"  {r['model']:<26} ADR={adr:.4f} ADR+w={adrw:.4f} GADR={gadr:.4f} GADR+w={gadrw:.4f} "
              f"A+w>=1={float(r['group_level_ADR_with_witness_at_least_1_pair_success']):.3f} "
              f"A+w>=10={float(r['group_level_ADR_with_witness_at_least_10_pair_success']):.3f} status={st}{flag}")
    for r in rows:
        if float(r["ADR_with_witness"]) > float(r["ADR"]) + 1e-12:
            bad += 1; print("  N-slice violation:", r["model"], r["slice_value"])
    if slug == "2sat":
        print("  two-sided (ADR+wu) overall rows:")
        for r in ov:
            wu = r["ADR_with_two_sided_witness"]
            if wu == "":
                print(f"    {r['model']:<32} ADR+wu=--  (no follow-up records)")
                continue
            wu = float(wu); w = float(r["ADR_with_witness"])
            if wu > w + 1e-12:
                bad += 1; flag = "  <-- VIOLATION (wu > w)"
            else:
                flag = ""
            print(f"    {r['model']:<32} ADR+w={w:.4f} ADR+wu={wu:.4f} GADR+wu={float(r['GADR_with_two_sided_witness']):.4f} "
                  f"neg-avail={r['num_pairs_negative_witness_available']} neg-verified={r['num_pairs_negative_witness_verified']} "
                  f"A+wu>=1={float(r['group_level_ADR_with_two_sided_witness_at_least_1_pair_success']):.3f} "
                  f"A+wu>=10={float(r['group_level_ADR_with_two_sided_witness_at_least_10_pair_success']):.3f}{flag}")
        for r in rows:
            wu = r["ADR_with_two_sided_witness"]
            if wu != "" and float(wu) > float(r["ADR_with_witness"]) + 1e-12:
                bad += 1; print("  N-slice two-sided violation:", r["model"], r["slice_value"])
    val = list(csv.DictReader(open(root / slug / "validation_against_archived.csv")))
    wv = [v for v in val if "witness" in v["metric"] and "two_sided" not in v["metric"]]
    tv = [v for v in val if "two_sided" in v["metric"]]
    if tv:
        tm = [v for v in tv if v["matches_within_1e-12"] != "True"]
        print(f"  two-sided-vs-archived comparisons: {len(tv)}, mismatches: {len(tm)}")
        for v in tm[:8]:
            print(f"    {v['model']:<26} {v['slice_type']}={v['slice_value']:<4} {v['metric']:<36} new={v['new_value'][:8]} archived={v['archived_value'][:8]}")
    mism = [v for v in wv if v["matches_within_1e-12"] != "True"]
    print(f"  witness-vs-archived comparisons: {len(wv)}, mismatches: {len(mism)}")
    for v in mism[:12]:
        print(f"    {v['model']:<26} {v['slice_type']}={v['slice_value']:<4} {v['metric']:<28} new={v['new_value'][:8]} archived={v['archived_value'][:8]} diff={v['absolute_difference'][:8]}")
    if len(mism) > 12: print(f"    ... {len(mism)-12} more")
print("BOUND VIOLATIONS:", bad)
sys.exit(1 if bad else 0)
