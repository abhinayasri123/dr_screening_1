import sqlite3
import json
from pathlib import Path

db_path = Path(r"C:\Users\VIJAY\.gemini\antigravity\scratch\dr_screening\dr_screening.db")
conn = sqlite3.connect(db_path)
conn.row_factory = sqlite3.Row
c = conn.cursor()

print("=" * 70)
print("1. CLINICAL REVIEWS IN DB (Total: %d)" % c.execute("SELECT count(*) FROM clinical_reviews").fetchone()[0])
print("=" * 70)
reviews = c.execute("SELECT * FROM clinical_reviews ORDER BY review_timestamp DESC").fetchall()
for r in reviews[:15]:
    print(dict(r))

print("\n" + "=" * 70)
print("2. PREDICTIONS IN DB (Total: %d)" % c.execute("SELECT count(*) FROM predictions").fetchone()[0])
print("=" * 70)
preds = c.execute("SELECT id, patient_id, image_filename, image_path, prediction_mode, predicted_class, confidence, review_status, created_at FROM predictions ORDER BY created_at DESC").fetchall()
for p in preds[:25]:
    print(dict(p))

print("\n" + "=" * 70)
print("3. REVIEW TIMESTAMPS GROUPING (Checking if multiple reviews were created together)")
print("=" * 70)
grouped_reviews = c.execute("""
    SELECT review_timestamp, count(*) as cnt, GROUP_CONCAT(prediction_id) as pids, GROUP_CONCAT(decision) as decs
    FROM clinical_reviews
    GROUP BY review_timestamp
    HAVING cnt > 1
    ORDER BY cnt DESC
""").fetchall()
for gr in grouped_reviews:
    print(f"Timestamp {gr['review_timestamp']}: {gr['cnt']} reviews! Prediction IDs: {gr['pids']}")

print("\n" + "=" * 70)
print("4. AUDIT EVENTS IN DB (Checking recent actions)")
print("=" * 70)
audits = c.execute("SELECT * FROM audit_events ORDER BY created_at DESC LIMIT 20").fetchall()
for a in audits:
    print(dict(a))
