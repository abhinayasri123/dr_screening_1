import urllib.request
import urllib.error
import json
import io
import os
import sys
from pathlib import Path

BASE_URL = "http://127.0.0.1:8000"
PROJECT_ROOT = Path(r"C:\Users\VIJAY\.gemini\antigravity\scratch\dr_screening")

def test_full_user_journey():
    print("=" * 65)
    print("TESTING FULL USER JOURNEY: NEW SCREENING -> REVIEW -> XAI -> REPORTS -> REGISTRY")
    print("=" * 65)

    # 1. Login
    login_data = json.dumps({"email": "clinician@dr-screening.com", "password": "clinician123"}).encode()
    req = urllib.request.Request(f"{BASE_URL}/api/v1/auth/login", data=login_data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req) as res:
        token = json.loads(res.read().decode())["access_token"]
    headers = {"Authorization": f"Bearer {token}"}
    print("[1] Logged in as clinician.")

    # 2. Get first patient
    p_req = urllib.request.Request(f"{BASE_URL}/api/v1/patients?limit=1", headers=headers)
    with urllib.request.urlopen(p_req) as res:
        patient = json.loads(res.read().decode())[0]
    print(f"[2] Selected patient: {patient['patient_code']} ({patient['full_name']})")

    # 3. Submit New Screening
    sample_img_path = PROJECT_ROOT / "data" / "retinal" / "test" / "2" / "demo_cls2_test_000.png"
    with open(sample_img_path, "rb") as f:
        img_bytes = f.read()

    boundary = "----WebKitFormBoundaryDRValidation2026"
    body = io.BytesIO()
    body.write(f"--{boundary}\r\nContent-Disposition: form-data; name=\"patient_id\"\r\n\r\n{patient['id']}\r\n".encode())
    body.write(f"--{boundary}\r\nContent-Disposition: form-data; name=\"mode\"\r\n\r\nimage_only\r\n".encode())
    body.write(f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"patient_fundus.png\"\r\nContent-Type: image/png\r\n\r\n".encode())
    body.write(img_bytes)
    body.write(f"\r\n--{boundary}--\r\n".encode())

    pred_req = urllib.request.Request(
        f"{BASE_URL}/api/v1/predictions",
        data=body.getvalue(),
        headers={**headers, "Content-Type": f"multipart/form-data; boundary={boundary}"}
    )
    with urllib.request.urlopen(pred_req) as res:
        pred_res = json.loads(res.read().decode())
    
    pred_id = pred_res["id"]
    print(f"[3] New Screening Created: Prediction ID = {pred_id}")
    print(f"    AI Diagnosis: Grade {pred_res['predicted_class']} ({pred_res['predicted_label']}), Confidence = {pred_res['confidence'] * 100:.1f}%")

    # 4. Check Clinician Review Queue
    rev_req = urllib.request.Request(f"{BASE_URL}/api/v1/reviews/pending", headers=headers)
    with urllib.request.urlopen(rev_req) as res:
        pending_list = json.loads(res.read().decode())
    
    pending_ids = [p["id"] for p in pending_list]
    assert pred_id in pending_ids, f"Prediction {pred_id} not found in Review Queue!"
    print(f"[4] Verified in Clinician Review Queue: Found {len(pending_list)} pending cases (includes new case).")

    # 5. Clinician Approves the Screening
    review_payload = json.dumps({
        "prediction_id": pred_id,
        "clinician_grade": pred_res["predicted_class"],
        "decision": "APPROVED",
        "clinician_notes": "Retinal fundus image reviewed and verified by attending clinician.",
        "final_diagnosis": pred_res["predicted_label"]
    }).encode()
    review_post = urllib.request.Request(
        f"{BASE_URL}/api/v1/reviews",
        data=review_payload,
        headers={**headers, "Content-Type": "application/json"}
    )
    with urllib.request.urlopen(review_post) as res:
        review_res = json.loads(res.read().decode())
    print(f"[5] Clinician Review Submitted: Decision = {review_res.get('decision')} | Review ID = {review_res.get('id')}")

    # 6. Check Clinical Reports
    rep_req = urllib.request.Request(f"{BASE_URL}/api/v1/screenings?limit=10", headers=headers)
    with urllib.request.urlopen(rep_req) as res:
        reports_list = json.loads(res.read().decode())
    
    matched_rep = next((r for r in reports_list if r.get("patient_id") == patient["id"]), None)
    assert matched_rep is not None, "Newly reviewed screening not found in Clinical Reports!"
    print(f"[6] Verified in Clinical Reports: Case {matched_rep['case_number']} | Status = {matched_rep['status']} | Grade = {matched_rep['predicted_grade']}")

    # 7. Check Explainability (XAI) & Grad-CAM Generation
    xai_req = urllib.request.Request(f"{BASE_URL}/api/v1/screenings/{matched_rep['id']}", headers=headers)
    with urllib.request.urlopen(xai_req) as res:
        xai_detail = json.loads(res.read().decode())
    
    assert len(xai_detail.get("images", [])) > 0, "No fundus image in XAI detail!"
    print(f"[7] Verified in Explainability (XAI): Case {xai_detail['encounter']['case_number']} has retinal image.")

    gradcam_post = urllib.request.Request(
        f"{BASE_URL}/api/v1/screenings/{matched_rep['id']}/gradcam?target_class={pred_res['predicted_class']}",
        data=b"{}",
        headers={**headers, "Content-Type": "application/json"}
    )
    with urllib.request.urlopen(gradcam_post) as res:
        gradcam_res = json.loads(res.read().decode())
    assert gradcam_res["status"] == "SUCCESS", f"Grad-CAM generation failed: {gradcam_res}"
    print(f"    Grad-CAM Generated Successfully: {gradcam_res['gradcam_url']}")

    # 8. Check Model Registry
    admin_req = urllib.request.Request(f"{BASE_URL}/api/v1/models", headers=headers)
    with urllib.request.urlopen(admin_req) as res:
        models = json.loads(res.read().decode())
    assert len(models) >= 2, "Expected at least 2 models in registry!"
    print(f"[8] Verified Model Registry: {len(models)} models registered.")
    for m in models:
        print(f"    - {m['name']} (Active: {m['is_active']})")

    print("\n" + "=" * 65)
    print("END-TO-END CLINICAL USER JOURNEY VERIFIED AND PASSED!")
    print("=" * 65)

if __name__ == "__main__":
    test_full_user_journey()
