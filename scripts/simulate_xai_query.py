import sqlite3
from pathlib import Path

BASE_DIR = Path("C:/Users/VIJAY/.gemini/antigravity/scratch/dr_screening")
DB_PATH = BASE_DIR / "dr_screening.db"

def simulate_query():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    
    print("--- 1. Querying Screening Encounters ---")
    cur.execute("""
        SELECT e.id, e.case_number, e.patient_id, e.status, e.encounter_date,
               p.patient_code, p.full_name
        FROM screening_encounters e
        LEFT JOIN patients p ON e.patient_id = p.id
        ORDER BY e.encounter_date DESC
        LIMIT 20
    """)
    rows = cur.fetchall()
    print(f"Found {len(rows)} encounters:")
    
    eligible_for_xai = []
    
    for r in rows:
        enc_id, case_num, pat_id, status, enc_date, pat_code, full_name = r
        
        # Check retinal images for this encounter
        cur.execute("""
            SELECT id, file_path, is_gradable, blur_score
            FROM retinal_images
            WHERE encounter_id = ?
            ORDER BY uploaded_at DESC LIMIT 1
        """, (enc_id,))
        img = cur.fetchone()
        
        # Check predictions for this encounter
        cur.execute("""
            SELECT id, image_path, gradcam_path, predicted_class, predicted_label, confidence
            FROM predictions
            WHERE encounter_id = ?
            ORDER BY created_at DESC LIMIT 1
        """, (enc_id,))
        pred = cur.fetchone()
        
        # If no prediction by encounter_id, check by image_path
        if not pred and img:
            cur.execute("""
                SELECT id, image_path, gradcam_path, predicted_class, predicted_label, confidence
                FROM predictions
                WHERE image_path = ?
                ORDER BY created_at DESC LIMIT 1
            """, (img[1],))
            pred = cur.fetchone()
            
        print(f"Case {case_num}: Status={status}, HasImage={img is not None}, HasPred={pred is not None}")
        if img or pred:
            eligible_for_xai.append({
                "enc_id": enc_id,
                "case_num": case_num,
                "patient_name": full_name,
                "image_path": img[1] if img else (pred[1] if pred else None),
                "gradcam_path": pred[2] if pred else None,
                "pred_class": pred[3] if pred else None,
                "pred_label": pred[4] if pred else None
            })
            
    print(f"\nTotal eligible for XAI: {len(eligible_for_xai)}")
    for x in eligible_for_xai[:5]:
        print(" ", x)
        
    conn.close()

if __name__ == "__main__":
    simulate_query()
