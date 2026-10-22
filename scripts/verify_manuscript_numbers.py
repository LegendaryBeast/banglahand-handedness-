#!/usr/bin/env python3
"""Check that the manuscripts agree with the saved results (42-writer run).

PRL: (1) regenerate the macro file from the results and require it to be identical to the one the paper uses;
(2) every \\v macro used in the text is defined; (3) the inlined table rows equal the generated table files.
DiB: dataset counts quoted in the text equal those recomputed from dataset/metadata/images.csv.
"""
import json, re, subprocess, sys, tempfile
from pathlib import Path
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
RUN = "results/final42_2026-10-19"
ok = True
with tempfile.TemporaryDirectory() as d:
    fresh = Path(d) / "m.tex"
    subprocess.run([sys.executable, str(ROOT / "scripts/make_prl_macros.py"), "--run", RUN, "--output", str(fresh)], check=True, capture_output=True)
    used = ROOT / "manuscripts/prl/figures/results_macros.tex"
    same = fresh.read_text().splitlines()[1:] == used.read_text().splitlines()[1:]
    print("PRL macros regenerated from results identical to the paper's:", same); ok &= same
tex = (ROOT / "manuscripts/prl/prl_manuscript.tex").read_text()
defined = set(re.findall(r"\\newcommand\{\\(v\w+)\}", used.read_text()))
missing = sorted(set(re.findall(r"\\(v[A-Z]\w*)", tex)) - defined)
print("PRL macros used but undefined:", missing or "none"); ok &= not missing
for f in ["table_main.tex", "table_pitfalls.tex", "table_text.tex"]:
    rows = (ROOT / "manuscripts/prl/figures" / f).read_text().strip().splitlines()
    inl = all(r in tex for r in rows); print(f"PRL {f} rows inlined unchanged:", inl); ok &= inl
img = pd.read_csv(ROOT / "dataset/metadata/images.csv"); dib = (ROOT / "manuscripts/data_in_brief/dib_manuscript.tex").read_text()
nt = int(img.text_truncated_at_edge.notna().sum())
ntl = int((img.text_truncated_at_edge.notna() & (img.hand == "left")).sum()); ntr = nt - ntl
NUM = {2: "two", 13: "Thirteen"}
M = dict(l.split(" = ") for l in [x.replace("\\newcommand{\\v", "").replace("}{", " = ").rstrip("}") for x in used.read_text().splitlines()[1:]])
au = json.loads((ROOT / "audit/geometric/summary.json").read_text())
gp = pd.read_csv(ROOT / "audit/geometric/geometric_pairs.csv.gz"); sw = gp[gp.kind == "same_writer"]
split = sw.a.str.contains("L_p14_|L_p21_") & sw.a.str.contains("para07") & sw.b.str.contains("para08")
within_max = int(sw[~split].inliers.max()); lo, hi = (f"{v:,}" for v in au["unreleased_P19_vs_R_p6_inliers"])
facts = {f"{len(img)} smartphone photographs": True, f"from {img.writer_id.nunique()} writers": True,
         f"{img[img.hand=='left'].writer_id.nunique()} left-handed and {img[img.hand=='right'].writer_id.nunique()} right-handed": True,
         f"{(img.condition=='shared').sum()} shared-prompt and {(img.condition!='shared').sum()} writer-specific images": True,
         f"{NUM.get(nt, nt)} images ({ntl} left, {ntr} right)": True,
         f"about {round(img.size_bytes.sum()/1e6)}\\,MB": True,
         f"never exceeded {au['same_prompt_different_writers_max_inliers']} RANSAC inliers": True,
         f"scored at most {au['same_prompt_different_writers_max_inliers']} inliers": True,
         f"within a writer at most {within_max}, except": True,
         f"scored {lo}--{hi} on every paragraph": True,
}
for k in facts:
    f = k in dib; print("DiB:", k, "->", f); ok &= f
print("ALL CONSISTENT" if ok else "INCONSISTENCIES FOUND"); sys.exit(0 if ok else 1)
