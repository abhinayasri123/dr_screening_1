import sys
import os
from pathlib import Path

BASE_DIR = Path("C:/Users/VIJAY/.gemini/antigravity/scratch/dr_screening")
sys.path.insert(0, str(BASE_DIR))
os.chdir(str(BASE_DIR))

import sqlite3

def analyze():
    conn = sqlite3.connect("dr_screening.db")
    cur = conn.cursor()
    
    print("--- Predictions breakdown ---")
    cur.execute("""
        SELECT p.id, p.patient_id, p.encounter_id, p.image_path, p.gradcam_path, p.created_at, p.predicted_class, p.predicted_label
        FROM predictions p
        ORDER BY p.created_at DESC
    """)
    preds = cur.fetchall()
    print(f"Total predictions: {len(preds)}")
    no_enc = [p for p in preds if not p[2]]
    print(f"Predictions without encounter_id: {len(no_enc)}")
    for p in no_enc[:5]:
        print("  Orphan prediction:", p[0], p[1], p[3], p[6], p[7])
        
    print("\n--- Encounters breakdown ---")
    cur.execute("""
        SELECT e.id, e.case_number, e.patient_id, e.status, e.encounter_date
        FROM screening_encounters e
        ORDER BY e.encounter_date DESC
    """)
    encs = cur.fetchall()
    print(f"Total encounters: {len(encs)}")
    for e in encs[:10]:
        # check if it has retinal images
        cur.execute("SELECT id, file_path FROM retinal_images WHERE encounter_id = ?", (e[0],))
        imgs = cur.fetchall()
        cur.execute("SELECT id, image_path, gradcam_path FROM predictions WHERE encounter_id = ?", (e[0],))
        ps = cur.fetchall()
        print(f"  Encounter {e[1]} ({e[3]}): imgs={len(imgs)}, preds={len(ps)}")
        
    conn.close()

if __name__ == "__main__":
    analyze()
