import sqlite3
import json
import hashlib
from pathlib import Path

def migrate_audit_table():
    db_path = Path(__file__).resolve().parent.parent / "dr_screening.db"
    conn = sqlite3.connect(str(db_path))
    c = conn.cursor()

    c.execute("PRAGMA table_info(audit_events)")
    cols = [r[1] for r in c.fetchall()]

    if 'sequence_num' not in cols:
        c.execute("ALTER TABLE audit_events ADD COLUMN sequence_num INTEGER")
    if 'outcome' not in cols:
        c.execute("ALTER TABLE audit_events ADD COLUMN outcome VARCHAR(20) DEFAULT 'SUCCESS'")
    if 'prev_hash' not in cols:
        c.execute("ALTER TABLE audit_events ADD COLUMN prev_hash VARCHAR(64)")
    if 'curr_hash' not in cols:
        c.execute("ALTER TABLE audit_events ADD COLUMN curr_hash VARCHAR(64)")

    conn.commit()

    # Backfill existing events chronologically
    c.execute("SELECT id, user_id, action, resource_type, resource_id, details, created_at, outcome FROM audit_events ORDER BY created_at ASC")
    rows = c.fetchall()
    print(f"[AuditMigration] Found {len(rows)} existing events.")

    prev_hash = "0" * 64
    for seq, (eid, uid, act, rtype, rid, det_json, cat, outc) in enumerate(rows, 1):
        outcome = outc or "SUCCESS"
        details = {}
        if det_json:
            try:
                details = json.loads(det_json) if isinstance(det_json, str) else det_json
            except Exception:
                details = {}
        
        canonical = {
            "prev_hash": prev_hash,
            "id": eid,
            "timestamp": str(cat),
            "user_id": uid or "",
            "action": act,
            "resource_type": rtype,
            "resource_id": rid or "",
            "outcome": outcome,
            "details": details
        }
        payload = json.dumps(canonical, sort_keys=True, separators=(',', ':'))
        curr_hash = hashlib.sha256(payload.encode('utf-8')).hexdigest()

        c.execute(
            """
            UPDATE audit_events 
            SET sequence_num = ?, outcome = ?, prev_hash = ?, curr_hash = ?
            WHERE id = ?
            """,
            (seq, outcome, prev_hash, curr_hash, eid)
        )
        prev_hash = curr_hash

    conn.commit()
    print(f"[AuditMigration] Migration and hash chain backfill complete. Total events: {len(rows)}, Latest hash: {prev_hash[:16]}...")
    conn.close()

if __name__ == "__main__":
    migrate_audit_table()
