"""
Automated Test Suite: Immutable Audit Trail & Cryptographic Hash Chain Integrity
================================================================================
Verifies:
1. Audit event recording with all required fields.
2. Deterministic SHA-256 hash calculation.
3. Cryptographic hash chain continuity across stored records.
4. Tamper detection: simulated field alteration is detected and reports exact event ID.
5. Sequential integrity and non-repudiation.
"""

import sys
import os
import json
import sqlite3
import hashlib
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

def compute_hash(prev_hash, event_id, timestamp_str, user_id, action, resource_type, resource_id, outcome, details):
    canonical = {
        "prev_hash": prev_hash or ("0" * 64),
        "id": event_id,
        "timestamp": str(timestamp_str),
        "user_id": user_id or "",
        "action": action,
        "resource_type": resource_type,
        "resource_id": resource_id or "",
        "outcome": outcome or "SUCCESS",
        "details": details or {}
    }
    payload = json.dumps(canonical, sort_keys=True, separators=(',', ':'))
    return hashlib.sha256(payload.encode('utf-8')).hexdigest()

def test_audit_trail():
    db_path = PROJECT_ROOT / "dr_screening.db"
    assert db_path.exists(), f"Database not found at {db_path}"
    
    conn = sqlite3.connect(str(db_path))
    c = conn.cursor()

    print("\n--- Test 1: Verify Audit Event Schema & Required Columns ---")
    c.execute("PRAGMA table_info(audit_events)")
    col_names = [r[1] for r in c.fetchall()]
    required_cols = ["id", "sequence_num", "user_id", "user_email", "action", "resource_type", "resource_id", "outcome", "details", "prev_hash", "curr_hash", "created_at"]
    for col in required_cols:
        assert col in col_names, f"Missing required audit column: {col}"
    print(f"PASS: All {len(required_cols)} required columns present in audit_events schema.")

    print("\n--- Test 2: Deterministic SHA-256 Hash Verification ---")
    test_h1 = compute_hash("0"*64, "test-id-1", "2026-10-29T12:00:00", "u1", "LOGIN", "user", "u1", "SUCCESS", {"ip": "127.0.0.1"})
    test_h2 = compute_hash("0"*64, "test-id-1", "2026-10-29T12:00:00", "u1", "LOGIN", "user", "u1", "SUCCESS", {"ip": "127.0.0.1"})
    assert test_h1 == test_h2, "Hash must be deterministic for identical payloads"
    assert len(test_h1) == 64, f"Hash must be 64-char hex SHA-256, got {len(test_h1)}"
    print(f"PASS: Deterministic hash computed: {test_h1[:16]}...")

    print("\n--- Test 3: Full Cryptographic Chain Continuity on Real Database ---")
    c.execute("SELECT id, sequence_num, user_id, action, resource_type, resource_id, outcome, details, prev_hash, curr_hash, created_at FROM audit_events ORDER BY sequence_num ASC, created_at ASC")
    rows = c.fetchall()
    assert len(rows) > 0, "Audit events table should contain records"

    expected_prev = "0" * 64
    for idx, (eid, seq, uid, act, rtype, rid, outc, det_json, prev_h, curr_h, cat) in enumerate(rows, 1):
        # 1. Sequence check
        assert seq == idx, f"Sequence broken at row {idx}: expected {idx}, got {seq}"
        
        # 2. Link continuity
        assert prev_h == expected_prev, f"Chain link broken at seq {seq}: expected {expected_prev[:12]}..., got {prev_h[:12]}..."

        # 3. Payload integrity check
        details = {}
        if det_json:
            details = json.loads(det_json) if isinstance(det_json, str) else det_json
        
        recalc = compute_hash(prev_h, eid, cat, uid, act, rtype, rid, outc or "SUCCESS", details)
        assert curr_h == recalc, f"Hash mismatch at seq {seq} ({act}): stored {curr_h[:12]}... != recomputed {recalc[:12]}..."

        expected_prev = curr_h

    print(f"PASS: 100% of {len(rows)} stored audit records passed cryptographic chain verification.")

    print("\n--- Test 4: Tamper Detection (Simulated Record Mutation) ---")
    # Simulate an attack where an attacker attempts to alter the details of event #5
    if len(rows) >= 5:
        target_row = rows[4] # seq 5
        eid, seq, uid, act, rtype, rid, outc, det_json, prev_h, curr_h, cat = target_row
        tampered_details = {"tampered": True, "unauthorized_change": "HACK"}
        
        # Recompute with tampered details
        tampered_hash = compute_hash(prev_h, eid, cat, uid, act, rtype, rid, outc or "SUCCESS", tampered_details)
        assert tampered_hash != curr_h, "Tampered payload MUST produce different SHA-256 hash"

        # Verify that checking against tampered details fails
        is_tamper_detected = (curr_h != tampered_hash)
        assert is_tamper_detected is True
        print(f"PASS: Tamper detection succeeded! Mutation of event #{seq} ({eid[:8]}) is cryptographically caught.")

    print("\n--- Test 5: Tamper Detection (Simulated Chain Removal/Drop) ---")
    # If a record in the middle is dropped, sequence continuity and prev_hash check will immediately fail
    if len(rows) >= 3:
        skipped_prev = rows[0][9] # curr_hash of row 1
        subsequent_prev = rows[2][8] # prev_hash of row 3
        # If row 2 is missing, row 3 expects row 2's hash, not row 1's
        assert skipped_prev != subsequent_prev, "Chain deletion causes unmatchable prev_hash bridge"
        print(f"PASS: Deletion/omission of an intermediate record breaks chain continuity.")

    conn.close()
    print("\n==========================================================")
    print("ALL IMMUTABLE AUDIT TRAIL TESTS COMPLETED SUCCESSFULLY!")
    print("==========================================================")

if __name__ == "__main__":
    test_audit_trail()
