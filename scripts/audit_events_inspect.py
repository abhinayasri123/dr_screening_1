import sqlite3
from pathlib import Path

db_path = Path(r"C:\Users\VIJAY\.gemini\antigravity\scratch\dr_screening\dr_screening.db")
conn = sqlite3.connect(db_path)
conn.row_factory = sqlite3.Row
c = conn.cursor()

print("ALL REVIEW AUDIT EVENTS:")
for row in c.execute("SELECT * FROM audit_events WHERE action LIKE '%REVIEW%' ORDER BY created_at DESC"):
    print(dict(row))
