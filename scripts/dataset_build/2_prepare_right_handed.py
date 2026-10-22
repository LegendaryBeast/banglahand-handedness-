#!/usr/bin/env python3
"""Step 2 of the dataset build: prepare the Right-Handed images (23 writers, 184 images).

Fixes all orientation anomalies (portrait, landscape, and inverted/flip mode)
so every image is saved in canonical, upright landscape orientation (4080 x 2296),
with EXIF orientation tags stripped, and standardized names paragraph_01.jpg .. paragraph_08.jpg.
Exports full metadata and SHA-256 checksums ready for experiments.
"""

from pathlib import Path
import hashlib
import io
import json
import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "camera_originals" / "Right-Handed"
OUT_BASE = ROOT / "dataset"
OUT_RH = OUT_BASE / "Right-Handed"
OUT_META = OUT_BASE / "metadata"

OUT_RH.mkdir(parents=True, exist_ok=True)
OUT_META.mkdir(parents=True, exist_ok=True)

PROMPT_NAMES = {
    1: ("shared", "S01", "Bangladesh"),
    2: ("shared", "S02", "Education"),
    3: ("shared", "S03", "Technology"),
    4: ("shared", "S04", "Environment"),
    5: ("shared", "S05", "Time"),
    6: ("shared", "S06", "Health"),
    7: ("writer_specific", "D07", "Writer Specific 1"),
    8: ("writer_specific", "D08", "Writer Specific 2"),
}

def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def pixel_sha256(im):
    rgb = im.convert("RGB")
    h = hashlib.sha256(f"{rgb.size}:RGB:".encode())
    h.update(rgb.tobytes())
    return h.hexdigest()

def main():
    print(f"Reading raw photos from {RAW_DIR}...")
    folders = sorted(RAW_DIR.iterdir(), key=lambda x: int(x.name[1:]) if x.name[1:].isdigit() else 99)
    
    metadata_rows = []
    total_processed = 0
    
    for p_dir in folders:
        if not p_dir.is_dir():
            continue
        writer_num = int(p_dir.name[1:])
        writer_id = f"R_p{writer_num}"
        dest_writer_dir = OUT_RH / p_dir.name
        dest_writer_dir.mkdir(parents=True, exist_ok=True)
        
        # Files sorted chronologically by timestamp
        raw_files = sorted(p_dir.glob("*.jpg"))
        assert len(raw_files) == 8, f"Expected 8 files for {p_dir.name}, found {len(raw_files)}"
        
        for para_idx, raw_file in enumerate(raw_files, 1):
            with Image.open(raw_file) as raw_im:
                exif = raw_im.getexif()
                exif_tag = exif.get(274, 1)
                camera_make = str(exif.get(271, "samsung")).strip()
                camera_model = str(exif.get(272, "Galaxy A17")).strip()
                extra = exif.get_ifd(34665) if 34665 in exif else {}
                capture_dt = str(extra.get(36867, exif.get(306, "")))
                
                raw_w, raw_h = raw_im.size
                # Canonical upright rotation:
                # Raw landscape (4080, 2296) was photographed inverted overhead -> rotate 180
                # Raw portrait (2296, 4080) was photographed 90 CW -> rotate 270
                rot_deg = 270 if raw_h > raw_w else 180
                upright_im = raw_im.rotate(rot_deg, expand=True).convert("RGB")
                
                # Verify upright landscape dimensions
                assert upright_im.width > upright_im.height, f"Not landscape: {raw_file}"
                assert upright_im.size == (4080, 2296), f"Unexpected size {upright_im.size} for {raw_file}"
                
            dest_file = dest_writer_dir / f"paragraph_{para_idx:02d}.jpg"
            # Save clean upright JPEG with no EXIF orientation tag
            upright_im.save(dest_file, "JPEG", quality=98, subsampling=0)
            
            # Post-save validation
            with Image.open(dest_file) as check_im:
                check_im.load()
                assert check_im.size == (4080, 2296), f"Verification failed for {dest_file}"
                assert check_im.getexif().get(274) is None, f"EXIF orientation tag found in {dest_file}"
                curated_sha = file_sha256(dest_file)
                curated_pix_sha = pixel_sha256(check_im)
                
            cond, p_code, topic = PROMPT_NAMES[para_idx]
            prompt_id = p_code if cond == "shared" else f"W_{writer_id}_{p_code}"
            
            metadata_rows.append({
                "image_id": f"{writer_id}_para{para_idx:02d}",
                "writer_id": writer_id,
                "hand": "right",
                "label": 0,
                "paragraph_number": para_idx,
                "condition": cond,
                "prompt_id": prompt_id,
                "paragraph_instance_id": f"{writer_id}_para{para_idx:02d}",
                "topic_gist": topic,
                "relative_path": f"Right-Handed/{p_dir.name}/paragraph_{para_idx:02d}.jpg",
                "sha256": curated_sha,
                "pixel_sha256": curated_pix_sha,
                "width": 4080,
                "height": 2296,
                "size_bytes": dest_file.stat().st_size,
                "capture_datetime": capture_dt,
                "camera_model": camera_model,
                "original_file": str(raw_file.relative_to(ROOT)),
                "raw_width": raw_w,
                "raw_height": raw_h,
                "exif_orientation_raw": exif_tag,
                "applied_rotation_deg": rot_deg,
            })
            total_processed += 1

    df_meta = pd.DataFrame(metadata_rows)
    csv_path = OUT_META / "rh_images.csv"
    df_meta.to_csv(csv_path, index=False)
    
    # Checksums file
    sums_path = OUT_BASE / "SHA256SUMS_RH.txt"
    with open(sums_path, "w") as f:
        for _, r in df_meta.iterrows():
            f.write(f"{r['sha256']}  {r['relative_path']}\n")
            
    print(f"\nSUCCESS: Prepared {total_processed} images for {len(folders)} right-handed writers.")
    print(f"Metadata exported to {csv_path}")
    print(f"Checksums saved to {sums_path}")
    
    # Summary report
    print("\nDataset Summary:")
    print(f"- Total Writers: {df_meta['writer_id'].nunique()} (R_p1 through R_p23)")
    print(f"- Total Images: {len(df_meta)} (8 paragraphs per writer)")
    print(f"- Standard Dimensions: {df_meta['width'].iloc[0]} x {df_meta['height'].iloc[0]} landscape")
    print(f"- Unique SHA256 Hashes: {df_meta['sha256'].nunique()}/{len(df_meta)}")
    print(f"- Rotations applied: {df_meta['applied_rotation_deg'].value_counts().to_dict()}")

if __name__ == "__main__":
    main()
