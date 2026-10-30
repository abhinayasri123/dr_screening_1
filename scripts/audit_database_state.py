import sqlite3
from pathlib import Path

db_path = Path(r"C:\Users\VIJAY\.gemini\antigravity\scratch\dr_screening\dr_screening.db")
conn = sqlite3.connect(db_path)
c = conn.cursor()

tables = [r[0] for r in c.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
print("Tables in DB:", tables)
print("=" * 60)
for t in tables:
    count = c.execute(f"SELECT count(*) FROM {t}").fetchone()[0]
    cols = [col[1] for col in c.execute(f"PRAGMA table_info({t})").fetchall()]
    print(f"{t} ({count} rows):")
    print(f"  Columns: {cols}")
    print()
