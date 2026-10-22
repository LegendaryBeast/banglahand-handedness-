#!/usr/bin/env python3
"""Step 5 of the dataset build: add the validation results to dataset/metadata (images are not modified).

Adds to images.csv: capture_session, file_processing, paragraph_identity_check, paragraph_instance_id,
fragment_note, text_truncated_at_edge, writer_text_note; writes writers.csv, excluded_writers.csv and SHA256SUMS.txt.
prompts.csv (Bangla text of S01-S06) is maintained by hand. Writing dates were not recorded.
"""
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
D = ROOT / "dataset"

TRUNC = {("R_p6", 1), ("L_p6", 3), ("L_p8", 3), ("R_p11", 3), ("L_p6", 4), ("L_p8", 4), ("R_p11", 4), ("R_p12", 4),
         ("R_p19", 4), ("L_p8", 5), ("R_p3", 5), ("L_p4", 6), ("L_p8", 6)}
NOTE = {("R_p4", 3): "writer ended S03 with the closing sentence of S04",
        ("R_p12", 6): "writer stopped after 'tai protidin' (last sentence incomplete on the page)",
        ("R_p2", 6): "page instruction line ('7. any one paragraph') copied below the paragraph"}
SPLIT = {"L_p14": "overlapping crops of one paragraph (about 3 lines appear in both)",
         "L_p21": "overlapping crops of one paragraph (one line appears in both)"}


def main():
    m = pd.read_csv(D / "metadata/images.csv")
    m["camera_model"] = m.camera_model.astype(str).str.replace("\x00", "").str.strip()
    m["capture_session"] = "2026-10-19"
    m["file_processing"] = "decoded, rotated upright per EXIF orientation, re-encoded as JPEG (quality 98, 4:4:4)"
    m["paragraph_identity_check"] = "verified (visual reading, 19 Oct 2026)"
    m["paragraph_instance_id"] = m.image_id
    m["fragment_note"] = ""
    for w, txt in SPLIT.items():
        sel = (m.writer_id == w) & m.paragraph_number.isin([7, 8])
        m.loc[sel, "paragraph_instance_id"] = f"{w}_para07-08"
        m.loc[sel, "fragment_note"] = [f"crop {k - 6} of 2: {txt}" for k in m.loc[sel, "paragraph_number"]]
    key = list(zip(m.writer_id, m.paragraph_number))
    m["text_truncated_at_edge"] = ["bottom" if k in TRUNC else "" for k in key]
    m["writer_text_note"] = [NOTE.get(k, "") for k in key]
    m["condition"] = m.condition.replace({"writer_specific": "writer_specific"})
    m.loc[m.paragraph_number >= 7, "topic_gist"] = "own text"
    m["prompt_id"] = [f"S{k:02d}" if k <= 6 else f"W_{w}_{k:02d}" for w, k in key]
    m["wn"] = m.writer_id.str.extract(r"_p(\d+)").astype(int)
    m = m.sort_values(["hand", "wn", "paragraph_number"]).drop(columns="wn")
    m.to_csv(D / "metadata/images.csv", index=False)

    w = (m.groupby("writer_id").agg(hand=("hand", "first"), label=("label", "first"), images=("image_id", "size"),
                                    first_capture=("capture_datetime", "min"), last_capture=("capture_datetime", "max"),
                                    truncated_images=("text_truncated_at_edge", lambda s: int((s != "").sum())))).reset_index()
    prov = pd.read_csv(D / "metadata/rh_reordering_provenance.csv").set_index("new_writer_id").original_raw_folder
    w["raw_folder"] = [("camera_originals/Right-Handed/" + prov[x]) if x in prov.index else
                       f"camera_originals/Left-Handed/P{x.split('_p')[1]}" for x in w.writer_id]
    w["wn"] = w.writer_id.str.extract(r"_p(\d+)").astype(int)
    w.sort_values(["hand", "wn"]).drop(columns="wn").to_csv(D / "metadata/writers.csv", index=False)
    pd.DataFrame([
        dict(raw_folder="camera_originals/Right-Handed/P19", decision="excluded",
             reason="all eight pages are the pages released as R_p6 (raw P7): 1,395-1,663 SIFT-RANSAC inliers per paragraph"),
        dict(raw_folder="camera_originals/Right-Handed/P3", decision="excluded (preparer's decision)",
             reason="writer-level outlier in stroke width and ink measures (max |z| = 4.1 of 44 writers); two photographs at 2296x4080 "
                    "with a different JPEG quantisation table; excluded to keep 21/21 writers. Not a duplicate of any writer. "
                    "Reported as a sensitivity analysis."),
    ]).to_csv(D / "metadata/excluded_writers.csv", index=False)
    (D / "SHA256SUMS.txt").write_text("".join(f"{r.sha256}  {r.relative_path}\n" for r in m.itertuples()))
    print(m.groupby("hand").size().to_dict(), "truncated", int((m.text_truncated_at_edge != "").sum()))


if __name__ == "__main__":
    main()
