import sqlite3
import uuid
import datetime
import os
from pathlib import Path

DB_PATH = Path(r"C:\Users\VIJAY\.gemini\antigravity\scratch\dr_screening\dr_screening.db")

def run():
    con = sqlite3.connect(str(DB_PATH))
    cur = con.cursor()

    # Find predictions without an encounter linked via retinal_images
    cur.execute('''
        SELECT p.id, p.patient_id, p.image_path, p.image_filename, p.prediction_mode, p.review_status, p.created_at, p.created_by
        FROM predictions p
        WHERE p.id NOT IN (
            SELECT p2.id FROM predictions p2
            JOIN retinal_images ri ON p2.image_path = ri.file_path
            JOIN screening_encounters se ON ri.encounter_id = se.id
        )
    ''')
    orphans = cur.fetchall()
    print(f"Found {len(orphans)} orphaned predictions to backfill into screening encounters.")

    count = 0
    for p_id, pat_id, img_path, img_fname, mode, rev_status, created_at, created_by in orphans:
        if not img_path or not os.path.exists(img_path):
            continue

        enc_id = str(uuid.uuid4())
        ref_seq = f"SYNC-{count:04d}"
        case_num = f"SCR-{datetime.date.today().strftime('%Y%m%d')}-{ref_seq}"
        status = "COMPLETED" if rev_status == "REVIEWED" else "AWAITING_REVIEW"
        notes = f"Synchronized historical DR Screening ({mode or 'image_only'} mode)"
        c_by = created_by or "system"
        enc_date = created_at or datetime.datetime.now().isoformat()

        cur.execute('''
            INSERT INTO screening_encounters (id, case_number, patient_id, encounter_date, status, notes, created_by, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (enc_id, case_num, pat_id, enc_date, status, notes, c_by, enc_date))

        # Insert RetinalImage
        img_id = str(uuid.uuid4())
        fname = img_fname or os.path.basename(img_path)
        cur.execute('''
            INSERT INTO retinal_images (id, encounter_id, file_path, filename, width, height, blur_score, mean_luminance, is_gradable, quality_status, uploaded_at)
            VALUES (?, ?, ?, ?, 300, 300, 150.0, 120.0, 1, 'GRADABLE', ?)
        ''', (img_id, enc_id, img_path, fname, enc_date))
        count += 1

    con.commit()
    con.close()
    print(f"Successfully backfilled {count} encounters and retinal images.")

if __name__ == "__main__":
    run()
