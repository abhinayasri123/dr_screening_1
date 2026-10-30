import os
import sqlite3
import numpy as np
from PIL import Image
import cv2
from pathlib import Path

PROJECT_ROOT = Path(r"C:\Users\VIJAY\.gemini\antigravity\scratch\dr_screening")
db_path = PROJECT_ROOT / "dr_screening.db"

conn = sqlite3.connect(db_path)
conn.row_factory = sqlite3.Row
c = conn.cursor()

def classify_fundus_image(img_path):
    if not os.path.exists(img_path):
        return "MISSING", "File does not exist on disk", (0, 0), {}
    
    file_size = os.path.getsize(img_path)
    if file_size == 0:
        return "CORRUPTED", "Zero byte empty file", (0, 0), {}
    
    try:
        with Image.open(img_path) as im:
            im.verify()
        with Image.open(img_path) as im:
            rgb = np.array(im.convert("RGB"))
    except Exception as e:
        return "CORRUPTED", f"Failed to decode: {e}", (0, 0), {}
    
    h, w, ch = rgb.shape
    if h == 0 or w == 0:
        return "CORRUPTED", "Invalid dimensions (0,0)", (w, h), {}
    
    # Calculate stats
    mean_r = float(np.mean(rgb[:, :, 0]))
    mean_g = float(np.mean(rgb[:, :, 1]))
    mean_b = float(np.mean(rgb[:, :, 2]))
    overall_mean = float(np.mean(rgb))
    overall_std = float(np.std(rgb))
    
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    
    corner_size = max(5, min(h, w) // 20)
    corners = [
        gray[:corner_size, :corner_size],
        gray[:corner_size, -corner_size:],
        gray[-corner_size:, :corner_size],
        gray[-corner_size:, -corner_size:]
    ]
    corner_mean = float(np.mean([np.mean(c_arr) for c_arr in corners]))
    
    stats = {
        "file_size": file_size,
        "resolution": f"{w}x{h}",
        "mean_rgb": [round(mean_r, 1), round(mean_g, 1), round(mean_b, 1)],
        "overall_mean": round(overall_mean, 1),
        "overall_std": round(overall_std, 1),
        "blur_score": round(blur_score, 1),
        "corner_mean": round(corner_mean, 1)
    }
    
    # Check for empty / near black / near white
    if overall_mean < 5.0 or overall_std < 3.0:
        return "EMPTY_OR_DARK", "Entirely black or uniform image with negligible variation", (w, h), stats
    if overall_mean > 250.0 and overall_std < 5.0:
        return "EMPTY_OR_WHITE", "Entirely white or blown-out blank image", (w, h), stats
        
    # Check for sub-resolution thumbnail
    if w < 224 or h < 224:
        if corner_mean > 160.0:
            return "UNGRADABLE_THUMBNAIL", f"Sub-resolution ({w}x{h} < 224x224) with white web border ({corner_mean:.1f})", (w, h), stats
        return "UNGRADABLE_POOR_QUALITY", f"Sub-resolution ({w}x{h} < 224x224)", (w, h), stats
        
    # Check for non-fundus
    # A fundus is predominantly red/orange/dark: mean_r > mean_b, and usually mean_r > mean_g
    # Synthetic demo data: mean_r > mean_b, green can be higher in severe DR classes (38, 61, 5)
    # But if blue > red or gray variance indicates non-eye image:
    if mean_b > mean_r + 20 and mean_b > mean_g + 20:
        return "NON_FUNDUS", "Blue channel dominates, uncharacteristic of retinal tissue", (w, h), stats
        
    # Check if circular aperture is present (corners dark, center bright)
    center_box = gray[h//4:3*h//4, w//4:3*w//4]
    center_mean = float(np.mean(center_box))
    if corner_mean > 180.0 and center_mean < 150.0:
        return "NON_FUNDUS_OR_INVERTED", "Periphery brighter than center (light border thumbnail)", (w, h), stats
        
    if blur_score < 40.0:
        return "POOR_QUALITY_BLURRED", f"Severe optical or motion blur (score {blur_score:.1f} < 40)", (w, h), stats
        
    return "VALID_FUNDUS", "Valid retinal fundus photograph with gradable characteristics", (w, h), stats

print("=" * 80)
print("AUDITING ALL PREDICTIONS TABLE IMAGES (Total: %d records)" % c.execute("SELECT count(*) FROM predictions").fetchone()[0])
print("=" * 80)

pred_rows = c.execute("""
    SELECT p.id, p.patient_id, pt.patient_code, p.image_path, p.image_filename, 
           p.predicted_class, p.predicted_label, p.confidence, p.review_status, p.created_at
    FROM predictions p
    LEFT JOIN patients pt ON p.patient_id = pt.id
    ORDER BY p.created_at ASC
""").fetchall()

status_counts = {}
flagged_for_manual_review = []
audit_records = []

for idx, r in enumerate(pred_rows):
    img_path = r["image_path"]
    cat, reason, dims, stats = classify_fundus_image(img_path)
    status_counts[cat] = status_counts.get(cat, 0) + 1
    
    record = {
        "index": idx + 1,
        "prediction_id": r["id"],
        "patient_id": r["patient_id"],
        "patient_code": r["patient_code"] or "N/A",
        "image_filename": r["image_filename"] or os.path.basename(img_path or ""),
        "image_path": img_path,
        "category": cat,
        "reason": reason,
        "dimensions": f"{dims[0]}x{dims[1]}",
        "predicted": f"Class {r['predicted_class']} ({r['predicted_label']})",
        "confidence": f"{r['confidence']*100:.1f}%" if r["confidence"] else "N/A",
        "review_status": r["review_status"],
        "stats": stats
    }
    audit_records.append(record)
    
    if cat != "VALID_FUNDUS":
        flagged_for_manual_review.append(record)
        print(f"[{idx+1}] FLAGGED ({cat}): PredID={r['id'][:8]} Patient={r['patient_code']} File={record['image_filename']}")
        print(f"     Reason: {reason} | Dims: {dims[0]}x{dims[1]} | Conf: {record['confidence']}")
    else:
        print(f"[{idx+1}] VALID: PredID={r['id'][:8]} Patient={r['patient_code']} File={record['image_filename']} ({dims[0]}x{dims[1]}) Conf={record['confidence']}")

print("\n" + "=" * 80)
print("PREDICTION IMAGES AUDIT SUMMARY:")
for cat, cnt in sorted(status_counts.items()):
    print(f"  {cat}: {cnt} records")
print(f"TOTAL CHECKED: {len(pred_rows)}")
print(f"TOTAL FLAGGED FOR MANUAL REVIEW: {len(flagged_for_manual_review)}")
print("=" * 80)

# Also check retinal_images table
print("\n" + "=" * 80)
print("AUDITING RETINAL_IMAGES TABLE (Total: %d records)" % c.execute("SELECT count(*) FROM retinal_images").fetchone()[0])
print("=" * 80)
ret_rows = c.execute("SELECT id, encounter_id, file_path, filename, width, height, blur_score, quality_status FROM retinal_images").fetchall()
ret_status_counts = {}
for idx, r in enumerate(ret_rows):
    cat, reason, dims, stats = classify_fundus_image(r["file_path"])
    ret_status_counts[cat] = ret_status_counts.get(cat, 0) + 1
    if cat != "VALID_FUNDUS":
        print(f"[{idx+1}] RetinalImage FLAGGED ({cat}): ID={r['id'][:8]} EncID={r['encounter_id'][:8]} File={r['filename']} Reason={reason}")

print("\nRETINAL_IMAGES TABLE SUMMARY:")
for cat, cnt in sorted(ret_status_counts.items()):
    print(f"  {cat}: {cnt} records")

# Also check files on disk in uploads/
print("\n" + "=" * 80)
print("AUDITING ALL FILES IN UPLOADS DIRECTORY")
print("=" * 80)
uploads_dir = PROJECT_ROOT / "uploads"
all_upload_files = list(uploads_dir.glob("*.*"))
print(f"Total files in {uploads_dir}: {len(all_upload_files)}")
upload_categories = {}
for f in all_upload_files:
    if f.name.startswith("gradcam_"):
        continue
    cat, reason, dims, stats = classify_fundus_image(str(f))
    upload_categories[cat] = upload_categories.get(cat, 0) + 1
    if cat != "VALID_FUNDUS":
        print(f"Upload file FLAGGED ({cat}): {f.name} Reason={reason}")

print("\nUPLOADS DIRECTORY ORIGINAL IMAGES SUMMARY:")
for cat, cnt in sorted(upload_categories.items()):
    print(f"  {cat}: {cnt} files")
