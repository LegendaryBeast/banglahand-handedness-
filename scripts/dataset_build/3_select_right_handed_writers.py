#!/usr/bin/env python3
"""Step 3 of the dataset build: select 21 right-handed writers (P1 through P21).

Removes R_p3 (scale/stroke outlier) and R_p19 (duplicate of R_p7),
reindexes remaining 21 RH writers to P1..P21,
re-generates metadata, checksums, and contact sheets for the balanced 42-writer cohort.
"""

from pathlib import Path
import hashlib
import shutil
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "camera_originals" / "Right-Handed"
OUT_BASE = ROOT / "dataset"
OUT_RH = OUT_BASE / "Right-Handed"
OUT_META = OUT_BASE / "metadata"
CONTACT_DIR = ROOT / "audit" / "contact_sheets_right_handed"

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

# The 21 retained folders in order (P3 and P19 excluded)
SELECTED_RAW_FOLDERS = [
    "P1", "P2", "P4", "P5", "P6", "P7", "P8", "P9", "P10",
    "P11", "P12", "P13", "P14", "P15", "P16", "P17", "P18",
    "P20", "P21", "P22", "P23"
]

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
    print("Rebalancing Right-Handed dataset to 21 writers (P1 through P21)...")
    print(f"Excluded: P3 (scale/stroke outlier) and P19 (duplicate of P7)")
    
    # Clean staging directory
    stage_dir = OUT_BASE / ".Right-Handed_staging"
    if stage_dir.exists():
        shutil.rmtree(stage_dir)
    stage_dir.mkdir(parents=True, exist_ok=True)
    
    metadata_rows = []
    provenance_rows = []
    
    font = ImageFont.load_default(size=18)
    CONTACT_DIR.mkdir(parents=True, exist_ok=True)
    # Clear old sheets
    for old_sheet in CONTACT_DIR.glob("sheet_P*.jpg"):
        old_sheet.unlink()

    for new_idx, raw_fld_name in enumerate(SELECTED_RAW_FOLDERS, 1):
        new_fld_name = f"P{new_idx}"
        new_writer_id = f"R_p{new_idx}"
        dest_writer_dir = stage_dir / new_fld_name
        dest_writer_dir.mkdir(parents=True, exist_ok=True)
        
        raw_p_dir = RAW_DIR / raw_fld_name
        raw_files = sorted(raw_p_dir.glob("*.jpg"))
        assert len(raw_files) == 8, f"Expected 8 files in {raw_fld_name}, found {len(raw_files)}"
        
        provenance_rows.append({
            "new_writer_id": new_writer_id,
            "new_folder": new_fld_name,
            "original_raw_folder": raw_fld_name,
            "original_capture_time_start": raw_files[0].stem,
            "original_capture_time_end": raw_files[-1].stem,
        })
        
        # Build contact sheet
        sheet = Image.new("RGB", (1600, 900), "#f3f4f6")
        draw = ImageDraw.Draw(sheet)
        draw.text((20, 15), f"Right-Handed Writer {new_fld_name} ({new_writer_id}) [orig: {raw_fld_name}] - Canonical Upright Landscape", fill="#111827", font=font)
        
        for para_idx, raw_file in enumerate(raw_files, 1):
            with Image.open(raw_file) as raw_im:
                exif = raw_im.getexif()
                exif_tag = exif.get(274, 1)
                camera_model = str(exif.get(272, "Galaxy A17")).strip()
                extra = exif.get_ifd(34665) if 34665 in exif else {}
                capture_dt = str(extra.get(36867, exif.get(306, "")))
                
                raw_w, raw_h = raw_im.size
                rot_deg = 270 if raw_h > raw_w else 180
                upright_im = raw_im.rotate(rot_deg, expand=True).convert("RGB")
                
                assert upright_im.size == (4080, 2296), f"Unexpected size {upright_im.size} for {raw_file}"
                
            dest_file = dest_writer_dir / f"paragraph_{para_idx:02d}.jpg"
            upright_im.save(dest_file, "JPEG", quality=98, subsampling=0)
            
            curated_sha = file_sha256(dest_file)
            curated_pix_sha = pixel_sha256(upright_im)
            
            cond, p_code, topic = PROMPT_NAMES[para_idx]
            prompt_id = p_code if cond == "shared" else f"W_{new_writer_id}_{p_code}"
            
            metadata_rows.append({
                "image_id": f"{new_writer_id}_para{para_idx:02d}",
                "writer_id": new_writer_id,
                "hand": "right",
                "label": 0,
                "paragraph_number": para_idx,
                "condition": cond,
                "prompt_id": prompt_id,
                "paragraph_instance_id": f"{new_writer_id}_para{para_idx:02d}",
                "topic_gist": topic,
                "relative_path": f"Right-Handed/{new_fld_name}/paragraph_{para_idx:02d}.jpg",
                "sha256": curated_sha,
                "pixel_sha256": curated_pix_sha,
                "width": 4080,
                "height": 2296,
                "size_bytes": dest_file.stat().st_size,
                "capture_datetime": capture_dt,
                "camera_model": camera_model,
                "original_file": str(raw_file.relative_to(ROOT)),
                "original_raw_folder": raw_fld_name,
                "applied_rotation_deg": rot_deg,
            })
            
            # Contact sheet thumbnail
            thumb_w, thumb_h = 370, 208
            thumb = upright_im.resize((thumb_w, thumb_h), Image.Resampling.LANCZOS)
            col = (para_idx - 1) % 4
            row = (para_idx - 1) // 4
            x = 20 + col * 395
            y = 55 + row * 270
            sheet.paste(thumb, (x, y + 25))
            draw.text((x, y + 2), f"Para {para_idx}: {raw_file.stem} ({new_writer_id})", fill="#374151", font=font)
            
        sheet.save(CONTACT_DIR / f"sheet_{new_fld_name}.jpg", quality=90)

    # Swap in the new Right-Handed directory
    if OUT_RH.exists():
        shutil.rmtree(OUT_RH)
    stage_dir.rename(OUT_RH)
    print(f"Replaced {OUT_RH} with 21 standardized folders (P1 through P21).")
    
    # Save provenance mapping
    df_prov = pd.DataFrame(provenance_rows)
    prov_path = OUT_META / "rh_reordering_provenance.csv"
    df_prov.to_csv(prov_path, index=False)
    print(f"Saved reordering provenance to {prov_path}")
    
    # Save RH metadata
    df_rh = pd.DataFrame(metadata_rows)
    rh_meta_path = OUT_META / "rh_images.csv"
    df_rh.to_csv(rh_meta_path, index=False)
    print(f"Saved RH metadata ({len(df_rh)} images) to {rh_meta_path}")
    
    # Combine with LH metadata
    lh_meta_path = OUT_META / "lh_images.csv"
    assert lh_meta_path.exists(), "LH metadata not found!"
    df_lh = pd.read_csv(lh_meta_path)
    
    df_combined = pd.concat([df_lh, df_rh], ignore_index=True)
    combined_meta_path = OUT_META / "images.csv"
    df_combined.to_csv(combined_meta_path, index=False)
    print(f"Saved combined metadata to {combined_meta_path}")
    
    # Combined SHA256SUMS.txt
    sums_path = OUT_BASE / "SHA256SUMS.txt"
    with open(sums_path, "w") as f:
        for _, r in df_combined.iterrows():
            f.write(f"{r['sha256']}  {r['relative_path']}\n")
    print(f"Saved combined SHA256SUMS.txt to {sums_path}")
    
    print("\n=== FINAL BALANCED DATASET SUMMARY ===")
    print(f"Left-Handed Writers:  {df_lh['writer_id'].nunique()} (L_p1 through L_p21) = {len(df_lh)} images")
    print(f"Right-Handed Writers: {df_rh['writer_id'].nunique()} (R_p1 through R_p21) = {len(df_rh)} images")
    print(f"Total Cohort:         {df_combined['writer_id'].nunique()} writers = {len(df_combined)} images")
    print(f"Images per writer:    8 paragraphs (all 4080 x 2296 landscape)")
    print(f"Unique SHA-256:       {df_combined['sha256'].nunique()}/{len(df_combined)}")

if __name__ == "__main__":
    main()
