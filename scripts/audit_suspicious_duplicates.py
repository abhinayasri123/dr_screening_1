import sqlite3
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(r"C:\Users\VIJAY\.gemini\antigravity\scratch\dr_screening")
conn = sqlite3.connect(PROJECT_ROOT / "dr_screening.db")
conn.row_factory = sqlite3.Row
c = conn.cursor()

print("=" * 80)
print("AUDITING DUPLICATE & SUSPICIOUS RAPID-FIRE SUBMISSIONS")
print("=" * 80)

# 1. Check patients with multiple predictions within 60 seconds
patient_preds = c.execute("""
    SELECT p.id as pred_id, p.patient_id, pt.patient_code, pt.full_name, 
           p.created_at, p.image_filename, p.predicted_class, p.confidence, p.is_demo_mode
    FROM predictions p
    LEFT JOIN patients pt ON p.patient_id = pt.id
    ORDER BY p.patient_id, p.created_at ASC
""").fetchall()

rapid_duplicates = []
by_patient = {}
for r in patient_preds:
    pid = r["patient_id"]
    by_patient.setdefault(pid, []).append(r)

for pid, preds in by_patient.items():
    if len(preds) > 1:
        for i in range(len(preds) - 1):
            t1_str = preds[i]["created_at"]
            t2_str = preds[i+1]["created_at"]
            try:
                t1 = datetime.fromisoformat(t1_str.replace("Z", "+00:00").split(".")[0])
                t2 = datetime.fromisoformat(t2_str.replace("Z", "+00:00").split(".")[0])
                diff_sec = abs((t2 - t1).total_seconds())
                if diff_sec <= 60:
                    rapid_duplicates.append({
                        "patient_id": pid,
                        "patient_code": preds[i]["patient_code"] or "UNKNOWN",
                        "patient_name": preds[i]["full_name"] or "UNKNOWN",
                        "pred_id_1": preds[i]["pred_id"],
                        "pred_id_2": preds[i+1]["pred_id"],
                        "time_diff_seconds": diff_sec,
                        "time_1": t1_str,
                        "time_2": t2_str,
                        "file_1": preds[i]["image_filename"],
                        "file_2": preds[i+1]["image_filename"],
                        "class_1": preds[i]["predicted_class"],
                        "class_2": preds[i+1]["predicted_class"],
                        "is_demo": bool(preds[i]["is_demo_mode"] or preds[i+1]["is_demo_mode"])
                    })
            except Exception as e:
                pass

print(f"Total rapid-fire duplicate candidate pairs (<=60s apart): {len(rapid_duplicates)}")
for dup in rapid_duplicates:
    print(f"\n[SUSPECTED DUPLICATE] Patient: {dup['patient_code']} ({dup['patient_name']})")
    print(f"  Time Difference: {dup['time_diff_seconds']:.1f}s | Demo Record: {dup['is_demo']}")
    print(f"  Pred 1: {dup['pred_id_1'][:8]} @ {dup['time_1']} (File: {dup['file_1']}, Class: {dup['class_1']})")
    print(f"  Pred 2: {dup['pred_id_2'][:8]} @ {dup['time_2']} (File: {dup['file_2']}, Class: {dup['class_2']})")
    
    # Analyze cause
    if "TEST" in dup["patient_code"] or "AUDIT" in dup["patient_code"] or "E2E" in dup["patient_code"] or "NOEHR" in dup["patient_code"]:
        cause = "Automated integration test execution (synthetic test fixture)"
    elif dup["time_diff_seconds"] <= 5 and dup["file_1"] == dup["file_2"]:
        cause = "Rapid double-click on UI Submit button (accidental duplicate submission)"
    else:
        cause = "Repeated image upload / re-test encounter within short interval"
    print(f"  Root Cause Assessment: {cause}")

# 2. Check for synthetic / demo patients
print("\n" + "=" * 80)
print("AUDITING PATIENT PROVENANCE (Real vs Demo/Synthetic Patients)")
print("=" * 80)
patients = c.execute("SELECT id, patient_code, full_name, created_at FROM patients").fetchall()
demo_patients = []
real_or_live_patients = []
for pt in patients:
    code = pt["patient_code"] or ""
    name = pt["full_name"] or ""
    is_test = any(k in code.upper() for k in ["TEST", "AUDIT", "E2E", "NOEHR", "P-TEST"]) or any(k in name.upper() for k in ["TEST", "AUDIT", "NO EHR"])
    if is_test:
        demo_patients.append(pt)
    else:
        real_or_live_patients.append(pt)

print(f"Total Patients in DB: {len(patients)}")
print(f"  Synthetic / Demo / Test Fixture Patients: {len(demo_patients)}")
print(f"  Genuine / Live Clinical Evaluation Patients: {len(real_or_live_patients)}")
for p in real_or_live_patients:
    print(f"    - {p['patient_code']}: {p['full_name']}")
