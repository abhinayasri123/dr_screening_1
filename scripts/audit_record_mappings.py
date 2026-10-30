import os
import sqlite3
from pathlib import Path

PROJECT_ROOT = Path(r"C:\Users\VIJAY\.gemini\antigravity\scratch\dr_screening")
conn = sqlite3.connect(PROJECT_ROOT / "dr_screening.db")
conn.row_factory = sqlite3.Row
c = conn.cursor()

print("=" * 80)
print("AUDITING ENCOUNTER TO PREDICTION & IMAGE MAPPINGS")
print("=" * 80)

encounters = c.execute("SELECT * FROM screening_encounters ORDER BY encounter_date DESC").fetchall()
print(f"Total screening encounters: {len(encounters)}")

matched_by_image = 0
unmatched_encounters = []

for enc in encounters:
    enc_id = enc["id"]
    case_no = enc["case_number"]
    pat_id = enc["patient_id"]
    
    # Get retinal images for encounter
    r_imgs = c.execute("SELECT * FROM retinal_images WHERE encounter_id = ?", (enc_id,)).fetchall()
    
    # Try to find matching prediction by image basename
    pred = None
    if r_imgs:
        img_base = os.path.basename(r_imgs[0]["file_path"])
        pred = c.execute("SELECT * FROM predictions WHERE image_path LIKE ? OR image_filename = ?", (f"%{img_base}%", img_base)).fetchone()
    
    # If not found by image, check if any prediction has this patient
    if pred:
        matched_by_image += 1
        rev = c.execute("SELECT * FROM clinical_reviews WHERE prediction_id = ?", (pred["id"],)).fetchone()
        rev_status = rev["decision"] if rev else "AWAITING_REVIEW"
        print(f"[MATCHED] Case {case_no} ({enc_id[:8]}) -> Pred {pred['id'][:8]} | File: {img_base} | Status: {enc['status']} | Rev: {rev_status}")
    else:
        unmatched_encounters.append(enc)
        print(f"[UNMATCHED] Case {case_no} ({enc_id[:8]}) -> Has {len(r_imgs)} retinal images, but no matching prediction by image path!")

print(f"\nEncounters matched to exact prediction by image file: {matched_by_image} / {len(encounters)}")
print(f"Unmatched encounters: {len(unmatched_encounters)}")
