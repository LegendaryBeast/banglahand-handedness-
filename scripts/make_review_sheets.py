#!/usr/bin/env python3
"""Contact sheets of the curated dataset for manual paragraph-identity review.

Writes (read-only on dataset images):
  <out>/paragraph_7_8_combined_left.jpg   all left-handed writers, paragraphs 7 and 8
  <out>/paragraph_7_8_combined_right.jpg  all right-handed writers, paragraphs 7 and 8
  <out>/paragraph_7_8_review.pdf          same tiles, 5 writers per page, larger
  <out>/all_paragraphs/<writer>.jpg        all eight paragraphs of each writer
"""

import argparse
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
FONT = "/System/Library/Fonts/Supplemental/Arial.ttf"


def font(size):
    try:
        return ImageFont.truetype(FONT, size)
    except OSError:
        return ImageFont.load_default()


NOTES = {}   # (writer_id, paragraph) -> short note; filled from metadata/images.csv in main()


def tile(path, width):
    with Image.open(path) as im:
        im = im.convert("RGB")
        return im.resize((width, round(im.height * width / im.width)), Image.LANCZOS)


def writer_order(meta, hand):
    ids = meta[meta.handedness == hand].writer_id.unique()
    return sorted(ids, key=lambda w: int(w.split("_p")[1]))


def pair_rows(meta, writers, tile_w, pad=14, label_h=34):
    """One row per writer: [paragraph_07 | paragraph_08] with captions."""
    rows = []
    for wid in writers:
        g = meta[meta.writer_id == wid].set_index("paragraph_number")
        tiles = [tile(ROOT / g.loc[k, "source_path"], tile_w) for k in (7, 8)]
        h = max(t.height for t in tiles)
        row = Image.new("RGB", (2 * tile_w + 3 * pad, h + label_h + pad), "white")
        d = ImageDraw.Draw(row)
        for i, (k, t) in enumerate(zip((7, 8), tiles)):
            x = pad + i * (tile_w + pad)
            note = NOTES.get((wid, k), "")
            d.text((x, 6), f"{wid}  paragraph_{k:02d}  ({g.loc[k, 'handedness']})  topic: {g.loc[k, 'topic_gist']}",
                   fill="black", font=font(20))
            if note:
                d.text((x + tile_w - 12 * len(note) - 10, 6), note, fill=(180, 0, 0), font=font(20))
            row.paste(t, (x, label_h))
        rows.append(row)
    return rows


def stack(rows, title, pad=14):
    th = 60
    w = max(r.width for r in rows)
    h = th + sum(r.height for r in rows) + pad
    sheet = Image.new("RGB", (w, h), (235, 235, 235))
    ImageDraw.Draw(sheet).text((pad, 16), title, fill="black", font=font(28))
    y = th
    for r in rows:
        sheet.paste(r, (0, y))
        y += r.height
    return sheet


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", required=True)
    ap.add_argument("--dataset", default="dataset", help="dataset folder (default: dataset)")
    args = ap.parse_args()
    out = ROOT / args.output
    (out / "all_paragraphs").mkdir(parents=True, exist_ok=True)
    meta = pd.read_csv(ROOT / args.dataset / "metadata/images.csv")
    for r in meta.fillna("").itertuples():      # quality notes recorded in the metadata
        note = "; ".join(x for x in [r.fragment_note.split(":")[0], "text cut at bottom" if r.text_truncated_at_edge else "",
                                     "see writer_text_note" if r.writer_text_note else ""] if x)
        if note:
            NOTES[(r.writer_id, r.paragraph_number)] = note
    meta = meta.rename(columns={"hand": "handedness"})
    meta["source_path"] = args.dataset + "/" + meta["relative_path"]
    meta["original_source_path"] = meta["original_file"]

    pdf_pages = []
    for hand in ("left", "right"):
        writers = writer_order(meta, hand)
        rows = pair_rows(meta, writers, tile_w=760)
        title = (f"{hand.upper()}-HANDED: curated paragraph_07 and paragraph_08 for all {len(writers)} writers "
                 f"(verified: all are writer-specific texts, none is a shared prompt; red = note)")
        stack(rows, title).save(out / f"paragraph_7_8_combined_{hand}.jpg", quality=88)
        big = pair_rows(meta, writers, tile_w=1100)
        for i in range(0, len(big), 5):
            pdf_pages.append(stack(big[i:i + 5], f"{hand.upper()}-HANDED writers {i + 1}-{min(i + 5, len(big))}: "
                                                 "paragraph_07 | paragraph_08"))
    pdf_pages[0].save(out / "paragraph_7_8_review.pdf", save_all=True, append_images=pdf_pages[1:], resolution=150)

    # Full eight-paragraph sheets per writer (4 x 2 grid)
    tw, pad, lh = 560, 10, 30
    for wid, g in meta.groupby("writer_id"):
        g = g.set_index("paragraph_number").sort_index()
        tiles = [tile(ROOT / g.loc[k, "source_path"], tw) for k in range(1, 9)]
        th = max(t.height for t in tiles)
        sheet = Image.new("RGB", (4 * tw + 5 * pad, 50 + 2 * (th + lh + pad)), "white")
        d = ImageDraw.Draw(sheet)
        d.text((pad, 10), f"{wid} ({g.handedness.iloc[0]}): curated paragraphs 1-8 "
                          "(1-6 shared prompts, 7-8 writer-specific)", fill="black", font=font(24))
        for k, t in zip(range(1, 9), tiles):
            r, c = divmod(k - 1, 4)
            x, y = pad + c * (tw + pad), 50 + r * (th + lh + pad)
            d.text((x, y + 4), f"paragraph_{k:02d}  [{g.loc[k, 'condition']}]", fill="black", font=font(20))
            sheet.paste(t, (x, y + lh))
        sheet.save(out / "all_paragraphs" / f"{wid}.jpg", quality=85)
    print("done", out)


if __name__ == "__main__":
    main()
