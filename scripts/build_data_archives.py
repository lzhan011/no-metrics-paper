#!/usr/bin/env python3
"""Maintainer tool: pack the prediction records and the dataset subset the
paper needs into gzip tarballs under data/ (each < 100 MB for GitHub).

Paths inside every tarball are relative to the repository root, so
`tar -xzf data/<name>.tar.gz -C <repo root>` recreates the evaluation trees
(SAT_Group_Evaluation/..., Graph_Colouring_Evaluation/...) that the
analysis code expects.  See scripts/unpack_data.sh.

Usage: python scripts/build_data_archives.py --source <workspace root> [--out data]
"""
import argparse, json, os, re, tarfile, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent

def rel_from(path, marker):
    path = str(path).replace("\\", "/")
    tok = "/" + marker + "/"
    return marker + "/" + path.split(tok, 1)[1] if tok in path else None

def add_tree(tar, src_root, rel_root):
    for dp, _, fs in os.walk(src_root):
        for f in sorted(fs):
            full = Path(dp) / f
            tar.add(full, arcname=str(Path(rel_root) / full.relative_to(src_root)))

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--source", required=True); ap.add_argument("--out", default=str(REPO / "data"))
    a = ap.parse_args(); src = Path(a.source).resolve(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    sat = src / "SAT_Group_Evaluation"; gce = src / "Graph_Colouring_Evaluation"
    manifest = {}
    def pack(name, entries):
        """entries: list of (absolute path, archive-relative path)"""
        path = out / f"{name}.tar.gz"
        with tarfile.open(path, "w:gz", compresslevel=6) as tar:
            for full, arc in entries:
                tar.add(full, arcname=arc, recursive=False)
        manifest[name] = {"files": len(entries), "bytes": path.stat().st_size}
        print(f"{name}: {len(entries)} files -> {path.stat().st_size/1e6:.1f} MB", flush=True)
    def walk_entries(root, marker):
        ents = []
        for dp, _, fs in os.walk(root):
            for f in sorted(fs):
                full = Path(dp) / f; ents.append((full, rel_from(full, marker)))
        return ents
    # --- 3-SAT commercial by_file trees (three paper models) ---
    rep = json.load(open(sat / "analysis/commercial_api_results_analysis/commercial_api_group_adr_report_RMC_caseB.json"))
    keys = ["chatgpt__gpt-5", "claude__claude-opus-4-7__claude_batch_groups_zero_shot_desc_adaptive_thinking_medium_effort", "deepseek__deepseek-reasoner"]
    ents = []
    for k in keys:
        for root in rep["input_roots_by_model"][k]:
            r = rel_from(root, "SAT_Group_Evaluation"); ents += walk_entries(src / r, "SAT_Group_Evaluation")
    pack("predictions_3sat_commercial", ents)
    # --- 2-SAT: one tarball per model (results + follow-up) ---
    rep2 = json.load(open(sat / "analysis/open_source_LLM_2SAT_analysis/open_source_LLM_2SAT_group_adr_with_wilson_and_cp_RMC_rmc_paper_open_vendor_case_b.json"))
    for m in rep2["models"]:
        ents = []
        for key in ("input_root", "unsat_clause_subset_followup_results"):
            p = m.get(key)
            if p:
                r = rel_from(p, "SAT_Group_Evaluation"); ents.append((src / r, r))
        short = re.sub(r"[^A-Za-z0-9]+", "_", m["model"].split("__")[2] if m["model"].count("__") >= 2 else m["model"])[:40].strip("_")
        pack(f"predictions_2sat_{short}", ents)
    # --- graph coloring: one tarball per results file ---
    rep3 = json.load(open(sat / "analysis/graph_coloring_results_analysis/open_source_LLM_3_Colouring_group_adr_with_wilson_and_cp_RMC_graph_paper_case_a.json"))
    for k, p in rep3["input_results_by_model"].items():
        r = rel_from(p, "Graph_Colouring_Evaluation")
        short = re.sub(r"[^A-Za-z0-9]+", "_", Path(r).parent.name.replace("cnf_", "").replace("_graph_coloring_cases_interleave_zero_shot_desc_allN_3_5_8_10_15_20_25_30_40_50_60", ""))[:50].strip("_")
        pack(f"predictions_gc_{short}", [(src / r, r)])
    # --- dataset subsets: only CNF instances referenced by the prediction records (+ sidecar .json meta) ---
    need = json.load(open(HERE / "needed_cnf.json"))
    roots = {"3sat": "SAT_Group_Evaluation/generate/output/min_edit_3sat_dataset", "2sat": "SAT_Group_Evaluation/generate/output/min_edit_2sat_dataset", "gc": "Graph_Colouring_Evaluation/generate/output/min_edit_graph_coloring_dataset"}
    for tag in roots:
        ents = []
        for t, rel in need:
            if t != tag: continue
            r = roots[tag] + "/" + rel; full = src / r
            if full.exists(): ents.append((full, r))
            side = full.with_suffix(".json")
            if side.exists(): ents.append((side, roots[tag] + "/" + str(Path(rel).with_suffix(".json"))))
        pack(f"dataset_{tag}_subset", ents)
    json.dump(manifest, open(out / "MANIFEST.json", "w"), indent=2)
    big = [k for k, v in manifest.items() if v["bytes"] > 95e6]
    print("DONE; over 95MB:", big)

if __name__ == "__main__":
    main()
