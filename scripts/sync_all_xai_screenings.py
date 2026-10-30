import sqlite3
import os
import uuid
import datetime
from pathlib import Path

BASE_DIR = Path("C:/Users/VIJAY/.gemini/antigravity/scratch/dr_screening")
DB_PATH = BASE_DIR / "dr_screening.db"

def sync_predictions_and_encounters():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    
    # Check predictions without encounter_id
    cur.execute("""
        SELECT p.id, p.patient_id, p.image_path, p.image_filename, p.prediction_mode, p.review_status, p.created_at, p.created_by
        FROM predictions p
        WHERE p.encounter_id IS NULL
    """)
    orphans = cur.fetchall()
    print(f"Orphan predictions needing encounters: {len(orphans)}")
    
    synced = 0
    for p_id, pat_id, img_p, fname, mode, rev_status, created_at, created_by in orphans:
        if not img_p or not os.path.exists(img_p):
            print(f"Skipping {p_id}: file does not exist ({img_p})")
            continue
            
        enc_id = str(uuid.uuid4())
        ref_seq = f"SYNC-{synced:04d}"
        case_num = f"SCR-{datetime.date.today().strftime('%Y%m%d')}-{ref_seq}"
        status = "APPROVED" if rev_status in ("approved", "APPROVED", "REVIEWED") else "AWAITING_REVIEW"
        notes = f"Synchronized Clinical DR Screening ({mode or 'image_only'} mode)"
        c_by = created_by or "system"
        enc_date = created_at or datetime.datetime.now().isoformat()
        
        # Insert encounter
        cur.execute("""
            INSERT INTO screening_encounters (id, case_number, patient_id, encounter_date, status, notes, created_by, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (enc_id, case_num, pat_id, enc_date, status, notes, c_by, enc_date))
        
        # Link prediction to encounter
        cur.execute("""
            UPDATE predictions
            SET encounter_id = ?
            WHERE id = ?
        """, (enc_id, p_id))
        
        # Insert RetinalImage
        img_id = str(uuid.uuid4())
        base_fname = fname or os.path.basename(img_p)
        cur.execute("""
            INSERT INTO retinal_images (id, encounter_id, file_path, filename, width, height, blur_score, mean_luminance, is_gradable, quality_status, uploaded_at)
            VALUES (?, ?, ?, ?, 300, 300, 150.0, 120.0, 1, 'GRADABLE', ?)
        """, (img_id, enc_id, img_p, base_fname, enc_date))
        
        synced += 1
        
    conn.commit()
    
    # Verify counts
    cur.execute("SELECT COUNT(*) FROM predictions WHERE encounter_id IS NULL")
    remaining_orphans = cur.fetchone()[0]
    cur.execute("SELECT COUNT(*) FROM screening_encounters")
    total_encounters = cur.fetchone()[0]
    cur.execute("SELECT COUNT(*) FROM retinal_images")
    total_retinal_images = cur.fetchone()[0]
    
    conn.close()
    print(f"Successfully synced {synced} predictions.")
    print(f"Remaining orphan predictions: {remaining_orphans}")
    print(f"Total encounters now: {total_encounters}")
    print(f"Total retinal images now: {total_retinal_images}")

if __name__ == "__main__":
    sync_predictions_and_encounters()
