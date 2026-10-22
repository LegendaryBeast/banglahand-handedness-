#!/usr/bin/env python3
"""Step 4 of the dataset build: fix the paragraph order of R_p10 in dataset/ (19 Oct 2026).

Visual reading: the photographs filed as paragraph 3,4,5,6 show prompts S05,S06,S03,S04.
Files are renamed (bytes unchanged) so that paragraph_k shows prompt S0k; metadata and
checksums are updated. Logged to audit/fix_log.tsv.
"""
from pathlib import Path
import pandas as pd
ROOT = Path(__file__).resolve().parents[2]
D = ROOT / "dataset"
FIX = {"R_p10": {3: 5, 4: 6, 5: 3, 6: 4}}          # new paragraph k <- currently filed paragraph v
FIELDS = ["sha256", "pixel_sha256", "width", "height", "size_bytes", "capture_datetime", "original_file",
          "raw_width", "raw_height", "exif_orientation_raw", "applied_rotation_deg", "original_raw_folder"]
log = []
for name in ["images.csv", "rh_images.csv"]:
    m = pd.read_csv(D / "metadata" / name)
    for w, mp in FIX.items():
        idx = {int(r.paragraph_number): i for i, r in m[m.writer_id == w].iterrows()}
        snap = {k: m.loc[idx[k], [f for f in FIELDS if f in m.columns]].copy() for k in idx}
        for k, v in mp.items():
            for f in snap[v].index:
                m.loc[idx[k], f] = snap[v][f]
    m.to_csv(D / "metadata" / name, index=False)
for w, mp in FIX.items():
    folder = D / "Right-Handed" / ("P" + w.split("_p")[1])
    tmp = {}
    for k, v in mp.items():
        src = folder / f"paragraph_{v:02d}.jpg"; t = folder / f".tmp_{k:02d}.jpg"; src.rename(t); tmp[k] = t
        log.append(f"rename\t{src.relative_to(ROOT)}\t{(folder / f'paragraph_{k:02d}.jpg').relative_to(ROOT)}")
    for k, t in tmp.items():
        t.rename(folder / f"paragraph_{k:02d}.jpg")
m = pd.read_csv(D / "metadata/images.csv")
(D / "SHA256SUMS.txt").write_text("".join(f"{r.sha256}  {r.relative_path}\n" for r in m.itertuples()))
rh = m[m.hand == "right"]; (D / "SHA256SUMS_RH.txt").write_text("".join(f"{r.sha256}  {r.relative_path}\n" for r in rh.itertuples()))
out = ROOT / "audit/fix_log.tsv"
out.write_text("action\tfrom\tto\n" + "\n".join(log) + "\n"); print("\n".join(log))
