import sys
import os
from pathlib import Path

BASE_DIR = Path("C:/Users/VIJAY/.gemini/antigravity/scratch/dr_screening")
sys.path.insert(0, str(BASE_DIR))
os.chdir(str(BASE_DIR))

import sqlite3

def inspect_xai_data():
    conn = sqlite3.connect("dr_screening.db")
    cur = conn.cursor()
    
    print("=== 1. Checking Predictions ===")
    cur.execute("""
        SELECT p.id, p.patient_id, p.encounter_id, p.image_path, p.gradcam_path, p.predicted_class, p.predicted_label, p.is_demo_mode
        FROM predictions p
    """)
    predictions = cur.fetchall()
    print(f"Total predictions: {len(predictions)}")
    
    print("\n=== 2. Checking Screening Encounters ===")
    cur.execute("""
        SELECT e.id, e.case_number, e.patient_id, e.status, e.encounter_date
        FROM screening_encounters e
    """)
    encounters = cur.fetchall()
    print(f"Total encounters: {len(encounters)}")
    
    print("\n=== 3. Checking Retinal Images ===")
    cur.execute("PRAGMA table_info(retinal_images)")
    print("Columns:", [c[1] for c in cur.fetchall()])
    cur.execute("""
        SELECT r.id, r.encounter_id, r.file_path, r.is_gradable
        FROM retinal_images r
    """)
    retinal_images = cur.fetchall()
    print(f"Total retinal images: {len(retinal_images)}")

    print("\n=== 4. Checking Matching Encounters and Predictions ===")
    cur.execute("""
        SELECT e.id, e.case_number, e.status, p.id, p.image_path, p.gradcam_path, r.id, r.file_path
        FROM screening_encounters e
        LEFT JOIN predictions p ON p.encounter_id = e.id
        LEFT JOIN retinal_images r ON r.encounter_id = e.id
    """)
    joined = cur.fetchall()
    print(f"Total joined rows: {len(joined)}")
    with_pred = [j for j in joined if j[3] is not None]
    with_img = [j for j in joined if j[4] is not None or j[7] is not None]
    print(f"Encounters with predictions: {len(with_pred)}")
    print(f"Encounters with images: {len(with_img)}")

    conn.close()

if __name__ == "__main__":
    inspect_xai_data()
