#!/usr/bin/env python3
"""Statistics behind the paper paragraph "Group metrics versus witness evidence".

W per domain (paper Table `tab:group-adr-overall`, default):
  3-SAT           -> ADR_with_witness            (ADR^{+w})
  2-SAT           -> ADR_with_witness            (ADR^{+w}; one-sided, as in
                                                  the paper)
  Graph coloring  -> ADR_with_witness            (ADR^{+a})
Pass --two-sided to use ADR_with_two_sided_witness (ADR^{+wu}) for 2-SAT
instead (rows without a follow-up run are then excluded).
Run after run_group_analysis.py; prints every number quoted in the paragraph.
"""
import sys
TWO_SIDED = "--two-sided" in sys.argv
import csv
import itertools
import statistics as st
from pathlib import Path

ROOT = Path(__file__).resolve().parent / "results"
DOMAINS = [("3sat", "3-SAT"), ("2sat", "2-SAT"), ("graph_coloring", "Graph coloring")]
DAGGER = {"Llama-Nemotron-Super-49B (r-on)", "QwQ-32B (think)", "Qwen3-14B (think)", "Qwen3-32B (think)"}
LABEL = ["GADR", "ADR", "SGADR", "SGADR_star",
         "group_level_ADR_at_least_1_pair_success", "group_level_ADR_at_least_10_pair_success"]


def wcol(slug):
    return "ADR_with_two_sided_witness" if (slug == "2sat" and TWO_SIDED) else "ADR_with_witness"


def awcol(slug, k):
    fam = "with_two_sided_witness" if (slug == "2sat" and TWO_SIDED) else "with_witness"
    return f"group_level_ADR_{fam}_at_least_{k}_pair_success"


def f(x):
    return None if x in ("", None) else float(x)


def rank_pairs(rows, a, b):
    c = d = t = 0
    for r1, r2 in itertools.combinations(rows, 2):
        da = f(r1[a]) - f(r2[a]); db = f(r1[b]) - f(r2[b])
        if abs(da) < 1e-12 or abs(db) < 1e-12:
            t += 1
        elif da * db > 0:
            c += 1
        else:
            d += 1
    return c, d, t


def main():
    all_slices = []
    for slug, dom in DOMAINS:
        rows = list(csv.DictReader(open(ROOT / slug / "group_metrics_primary_lambda1.csv")))
        for r in rows:
            r["_W"] = f(r[wcol(slug)]); r["_AW1"] = f(r[awcol(slug, 1)]); r["_AW10"] = f(r[awcol(slug, 10)]); r["_slug"] = slug
        ov = [r for r in rows if r["slice_type"] == "overall"]
        ns = [r for r in rows if r["slice_type"] == "N" and r["_W"] is not None]
        all_slices += ns
        print(f"### {dom}: {len(ov)} overall rows ({sum(r['_W'] is None for r in ov)} without W), {len(ns)} N-slices with W")
        for r in ov:
            if r["_W"] is None:
                print(f"  {r['model']:<32} W=--"); continue
            W = r["_W"]; adr = f(r["ADR"]); g = f(r["GADR"]); sg = f(r["SGADR"]); sgs = f(r["SGADR_star"])
            print(f"  {r['model']:<32} W={W:.3f} ADR-W={adr-W:+.3f} GADR-W={g-W:+.3f} SGADR-W={sg-W:+.3f} SGADR*-W={sgs-W:+.3f} "
                  f"A1={f(r['group_level_ADR_at_least_1_pair_success']):.3f} AW1={r['_AW1']:.3f} "
                  f"A10={f(r['group_level_ADR_at_least_10_pair_success']):.3f} AW10={r['_AW10']:.3f} W/ADR={W/adr if adr else float('nan'):.2f}")
        ovw = [r for r in ov if r["_W"] is not None]
        if len(ovw) >= 3:
            for m in LABEL:
                c, d, t = rank_pairs(ovw, m, "_W")
                print(f"  ordering vs W by {m:<44}: concordant {c} discordant {d} tied {t} (of {c+d+t})")
    n = len(all_slices)
    print(f"### pooled N-slices with W defined: {n}")
    gap = [f(r["ADR"]) - r["_W"] for r in all_slices]
    print(f"  ADR-W gap: mean {st.mean(gap):.3f} median {st.median(gap):.3f} max {max(gap):.3f}; W<ADR in {sum(g>0 for g in gap)}/{n}, W==ADR in {sum(abs(g)<1e-12 for g in gap)}")
    for m in ["SGADR", "SGADR_star", "GADR"]:
        g2 = [f(r[m]) - r["_W"] for r in all_slices]
        closer = sum(1 for a, b in zip(gap, g2) if abs(b) < abs(a)); below = sum(1 for b in g2 if b < -1e-12)
        print(f"  {m}: mean gap to W {st.mean(g2):+.3f}; |gap| < ADR's in {closer}/{n}; below W in {below}/{n}")
    a10 = [(f(r["group_level_ADR_at_least_10_pair_success"]), r["_AW10"]) for r in all_slices]
    print(f"  A10>0 slices: {sum(1 for a,b in a10 if a>0)}; of these AW10=0: {sum(1 for a,b in a10 if a>0 and b==0)}")
    a1 = [(f(r["group_level_ADR_at_least_1_pair_success"]), r["_AW1"]) for r in all_slices]
    full = [b for a, b in a1 if a >= 1 - 1e-12]
    print(f"  A1=1.000 slices: {len(full)}; AW1 among them: min {min(full):.3f} median {st.median(full):.3f}")
    z = [r for r in all_slices if r["_W"] == 0]
    zpos = [r for r in z if f(r["ADR"]) > 0 and f(r["SGADR"]) > 0 and f(r["group_level_ADR_at_least_1_pair_success"]) > 0]
    print(f"  W=0 slices: {len(z)}; with ADR,SGADR,A1 all >0: {len(zpos)}; of those outside dagger 2-SAT rows: "
          f"{sum(1 for r in zpos if not (r['_slug']=='2sat' and r['model'] in DAGGER))} "
          f"(3-SAT {sum(r['_slug']=='3sat' for r in zpos)}, 2-SAT {sum(r['_slug']=='2sat' for r in zpos)}, GC {sum(r['_slug']=='graph_coloring' for r in zpos)})")
    if z:
        mx = max(zpos, key=lambda r: f(r["ADR"]))
        print(f"  max ADR among W=0 slices: {f(mx['ADR']):.3f} ({mx['domain']} {mx['model']} N={mx['slice_value']}, A1={f(mx['group_level_ADR_at_least_1_pair_success']):.3f})")
    for slug in ("3sat", "2sat"):
        ss = [r for r in all_slices if r["_slug"] == slug]
        print(f"  {slug}: W=0 & ADR>0 slices:", [(r["model"][:14], r["slice_value"], round(f(r["ADR"]), 3)) for r in ss if r["_W"] == 0 and f(r["ADR"]) > 0][:12])
    # 2-SAT specific: two-sided vs one-sided
    s2 = [r for r in all_slices if r["_slug"] == "2sat"]
    if s2:
        print("### 2-SAT: one-sided ADR+w vs two-sided ADR+wu on the same slices")
        w1 = [f(r["ADR_with_witness"]) for r in s2]; w2 = [r["_W"] for r in s2]
        print(f"  slices {len(s2)}; mean ADR+w {st.mean(w1):.3f}, mean ADR+wu {st.mean(w2):.3f}; ADR+wu==0 in {sum(x==0 for x in w2)}/{len(s2)}; ADR+w>0 & ADR+wu==0 in {sum(1 for a,b in zip(w1,w2) if a>0 and b==0)}")


if __name__ == "__main__":
    main()
