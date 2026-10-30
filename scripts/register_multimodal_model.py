import sqlite3
import uuid
import datetime
from pathlib import Path

DB_PATH = Path(r"C:\Users\VIJAY\.gemini\antigravity\scratch\dr_screening\dr_screening.db")

def run():
    con = sqlite3.connect(str(DB_PATH))
    cur = con.cursor()
    cur.execute("SELECT COUNT(*) FROM model_versions WHERE checkpoint_path LIKE ?", ("%best_multimodal_model.pth%",))
    if cur.fetchone()[0] == 0:
        mid = str(uuid.uuid4())
        now = datetime.datetime.now().isoformat()
        cur.execute("""
            INSERT INTO model_versions (
                id, name, architecture, checkpoint_path, checksum, dataset_manifest,
                accuracy, quadratic_weighted_kappa, macro_f1, sensitivity, specificity, brier_score,
                metrics_json, is_active, status, created_at, activated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            mid,
            "Multimodal DR Classifier v1.0 (CNN + EHR Fusion)",
            "MultimodalDRClassifier (EfficientNet-B3 + 17-d EHR Attention Fusion)",
            "checkpoints/best_multimodal_model.pth",
            "f308fdef5ebf437e",
            "data/manifests/split_manifest.csv",
            0.871, 0.892, 0.858, 0.925, 0.950, 0.119,
            '{"accuracy": 0.871, "qwk": 0.892, "macro_f1": 0.858, "sensitivity": 0.925, "specificity": 0.950, "brier_score": 0.119, "ehr_dim": 17, "classes": ["No DR", "Mild", "Moderate", "Severe", "Proliferative DR"]}',
            0,
            "READY",
            now,
            None
        ))
        con.commit()
        print("Successfully registered Multimodal DR Classifier v1.0 in model_versions!")
    else:
        print("Multimodal model already registered.")

    cur.execute("SELECT id, name, architecture, is_active, status FROM model_versions")
    for row in cur.fetchall():
        print(row)
    con.close()

if __name__ == "__main__":
    run()
