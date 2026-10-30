import os
import sqlite3
from pathlib import Path

PROJECT_ROOT = Path(r"C:\Users\VIJAY\.gemini\antigravity\scratch\dr_screening")
db_path = PROJECT_ROOT / "dr_screening.db"

conn = sqlite3.connect(db_path)
conn.row_factory = sqlite3.Row
c = conn.cursor()

print("=" * 70)
print("REPAIRING DATABASE IMAGE PATHS & LINKING ENCOUNTERS")
print("=" * 70)

# 1. Add encounter_id column to predictions if not present
cols = [col[1] for col in c.execute("PRAGMA table_info(predictions)").fetchall()]
if "encounter_id" not in cols:
    print("Adding encounter_id column to predictions table...")
    c.execute("ALTER TABLE predictions ADD COLUMN encounter_id VARCHAR(36)")
    conn.commit()
    print("encounter_id column added.")

# 2. Repair paths in predictions
pred_rows = c.execute("SELECT id, image_path, gradcam_path FROM predictions").fetchall()
pred_repaired_img = 0
pred_repaired_gradcam = 0

for r in pred_rows:
    pid = r["id"]
    ip = r["image_path"]
    gp = r["gradcam_path"]
    
    new_ip = None
    if ip and not os.path.exists(ip):
        base = os.path.basename(ip)
        target = PROJECT_ROOT / "uploads" / base
        if target.exists():
            new_ip = str(target)
            c.execute("UPDATE predictions SET image_path = ? WHERE id = ?", (new_ip, pid))
            pred_repaired_img += 1
            
    new_gp = None
    if gp and not os.path.exists(gp):
        gbase = os.path.basename(gp)
        gtarget = PROJECT_ROOT / "uploads" / gbase
        if gtarget.exists():
            new_gp = str(gtarget)
            c.execute("UPDATE predictions SET gradcam_path = ? WHERE id = ?", (new_gp, pid))
            pred_repaired_gradcam += 1

# 3. Repair paths in retinal_images
ret_rows = c.execute("SELECT id, file_path FROM retinal_images").fetchall()
ret_repaired = 0
for r in ret_rows:
    rid = r["id"]
    fp = r["file_path"]
    if fp and not os.path.exists(fp):
        base = os.path.basename(fp)
        target = PROJECT_ROOT / "uploads" / base
        if target.exists():
            c.execute("UPDATE retinal_images SET file_path = ? WHERE id = ?", (str(target), rid))
            ret_repaired += 1

conn.commit()
print(f"Repaired {pred_repaired_img} prediction image paths.")
print(f"Repaired {pred_repaired_gradcam} prediction Grad-CAM paths.")
print(f"Repaired {ret_repaired} retinal_images file paths.")

# 4. Link predictions to encounters by matching image basename
link_stmt = """
    UPDATE predictions
    SET encounter_id = (
        SELECT ri.encounter_id 
        FROM retinal_images ri 
        WHERE ri.file_path = predictions.image_path
        LIMIT 1
    )
    WHERE predictions.encounter_id IS NULL AND predictions.image_path IS NOT NULL
"""
c.execute(link_stmt)
conn.commit()

linked_count = c.execute("SELECT count(*) FROM predictions WHERE encounter_id IS NOT NULL").fetchone()[0]
print(f"Linked {linked_count} predictions directly to their ScreeningEncounter.")

print("=" * 70)
print("PATH REPAIR AND LINKING COMPLETED SUCCESSFULLY!")
print("=" * 70)
