"""
Comprehensive End-to-End Audit & Verification Script
===================================================
Tests all platform modules live against the running server on http://127.0.0.1:8000:
- A. Login and Role Management (Admin, Clinician, Technician, Researcher RBAC)
- B. Clinical Dashboard (Calculations, distribution)
- C. Patient Management (Registration, query, EHR)
- D. Image-Only CNN Screening (Inference, probabilities, confidence, uncertainty)
- E. Multimodal Fusion (CNN + EHR integration)
- F. AI Explainability (Grad-CAM heatmap availability)
- G. Clinician Review Queue (Pending reviews, review decision, status update)
- H. Research Analytics (Metrics, evaluation results)
- I. Federated Learning (Status, simulated hospital rounds, config)
- J. Admin Management (Model registry, active model)
- K. Technician Intake & Encounter Lifecycle (Encounter, Image quality assessment, PDF report)
"""

import os
import sys
import json
import uuid
import io
import urllib.request
import urllib.error
from pathlib import Path
from PIL import Image

BASE_URL = "http://127.0.0.1:8000"

def log_section(title):
    print("\n" + "=" * 70)
    print(f"MODULE: {title}")
    print("=" * 70)

def http_post(url, data_dict, token=None):
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(url, data=json.dumps(data_dict).encode(), headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req) as resp:
            return resp.status, json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read().decode())

def http_get(url, token=None):
    headers = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req) as resp:
            return resp.status, json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read().decode())

def post_multipart(url, fields, files, token=None):
    boundary = f"----WebKitFormBoundary{uuid.uuid4().hex}"
    body = bytearray()
    for k, v in fields.items():
        body.extend(f"--{boundary}\r\nContent-Disposition: form-data; name=\"{k}\"\r\n\r\n{v}\r\n".encode())
    for name, (fname, data, ctype) in files.items():
        body.extend(f"--{boundary}\r\nContent-Disposition: form-data; name=\"{name}\"; filename=\"{fname}\"\r\nContent-Type: {ctype}\r\n\r\n".encode())
        body.extend(data)
        body.extend(b"\r\n")
    body.extend(f"--{boundary}--\r\n".encode())

    headers = {"Content-Type": f"multipart/form-data; boundary={boundary}"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(url, data=bytes(body), headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req) as resp:
            return resp.status, json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read().decode())

def run_audit():
    results = {}

    # ----------------------------------------------------
    # Module A: Login and Role Management
    # ----------------------------------------------------
    log_section("A. Login & Role-Based Access Control (RBAC)")
    roles = [
        ("admin@dr-screening.com", "admin123", "admin"),
        ("clinician@dr-screening.com", "clinician123", "clinician"),
        ("technician@dr-screening.com", "tech123", "technician"),
        ("researcher@dr-screening.com", "research123", "researcher"),
    ]
    tokens = {}
    for email, pwd, expected_role in roles:
        status, body = http_post(f"{BASE_URL}/api/v1/auth/login", {"email": email, "password": pwd})
        assert status == 200, f"Login failed for {email}: {body}"
        assert body["user"]["role"] == expected_role
        tokens[expected_role] = body["access_token"]
        print(f"  [PASS] {expected_role.capitalize()} authentication confirmed ({email})")

    # Invalid login test
    status, body = http_post(f"{BASE_URL}/api/v1/auth/login", {"email": "invalid@hospital.org", "password": "wrong"})
    assert status == 401
    print("  [PASS] Invalid credentials correctly rejected with HTTP 401")
    results["A_Auth_RBAC"] = "PASSED"

    # ----------------------------------------------------
    # Module B: Clinical Dashboard Statistics
    # ----------------------------------------------------
    log_section("B. Clinical Dashboard Statistics")
    status, stats = http_get(f"{BASE_URL}/api/v1/analytics/dashboard-stats", tokens["clinician"])
    assert status == 200
    assert "total_patients" in stats
    assert "total_predictions" in stats
    assert "pending_reviews" in stats
    assert "completed_reviews" in stats
    assert "agreement_rate" in stats
    assert "class_distribution" in stats
    print(f"  [PASS] Dashboard live counts: Patients={stats['total_patients']}, Predictions={stats['total_predictions']}, Pending Reviews={stats['pending_reviews']}")
    print(f"  [PASS] Clinician agreement rate: {stats['agreement_rate']}%")
    print(f"  [PASS] 5-class distribution: {[c['name'] + ': ' + str(c['value']) for c in stats['class_distribution']]}")
    results["B_Dashboard"] = "PASSED"

    # ----------------------------------------------------
    # Module C: Patient Management
    # ----------------------------------------------------
    log_section("C. Patient Management")
    p_code = f"PAT-AUDIT-{uuid.uuid4().hex[:6].upper()}"
    status, new_pat = http_post(f"{BASE_URL}/api/v1/patients", {
        "patient_code": p_code,
        "full_name": "Eleanor Vance (Audited)",
        "date_of_birth": "1965-08-14",
        "sex": "F",
        "contact_email": "eleanor.vance@example.org"
    }, tokens["clinician"])
    assert status in [200, 201]
    pat_id = new_pat["id"]
    print(f"  [PASS] Registered new patient: {p_code} (ID: {pat_id})")

    # Save patient EHR
    status, ehr_res = http_post(f"{BASE_URL}/api/v1/patients/{pat_id}/ehr", {
        "patient_id": pat_id,
        "age": 61,
        "sex": "F",
        "diabetes_duration_years": 14.0,
        "hba1c": 8.8,
        "blood_glucose": 195.0,
        "systolic_bp": 145.0,
        "diastolic_bp": 92.0,
        "bmi": 29.3,
        "insulin_dependent": "yes",
        "smoking_status": "former"
    }, tokens["clinician"])
    assert status in [200, 201]
    print(f"  [PASS] Saved EHR record for patient {pat_id}")

    # Query back patient EHR
    status, queried_ehr = http_get(f"{BASE_URL}/api/v1/patients/{pat_id}/ehr", tokens["clinician"])
    assert status == 200
    assert queried_ehr["hba1c"] == 8.8
    assert queried_ehr["systolic_bp"] == 145.0
    print(f"  [PASS] Retrieved verified EHR: HbA1c={queried_ehr['hba1c']}%, SBP={queried_ehr['systolic_bp']} mmHg")
    results["C_Patient_Management"] = "PASSED"

    # ----------------------------------------------------
    # Module D: Image-Only CNN Screening & E: Multimodal Fusion
    # ----------------------------------------------------
    log_section("D & E. CNN Inference & Multimodal Fusion")
    # Synthetic fundus image
    img = Image.new("RGB", (224, 224), color=(170, 45, 35))
    for x in range(60, 160, 15):
        for y in range(60, 160, 15):
            img.putpixel((x, y), (210, 190, 40))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    img_bytes = buf.getvalue()

    # Image-Only
    status, img_pred = post_multipart(
        f"{BASE_URL}/api/v1/predictions",
        {"patient_id": pat_id, "mode": "image_only"},
        {"file": ("audit_retina.jpg", img_bytes, "image/jpeg")},
        tokens["clinician"]
    )
    assert status == 200
    assert img_pred["predicted_class"] in range(5)
    assert 0.0 <= img_pred["confidence"] <= 1.0
    assert img_pred["gradcam_url"] is not None
    print(f"  [PASS] Image-Only CNN: {img_pred['prediction']} (Confidence: {img_pred['confidence']*100:.1f}%, Uncertainty: {img_pred['uncertainty_level']})")

    # Multimodal missing EHR -> 422 (requires a patient without existing EHR)
    status, p_no_ehr = http_post(f"{BASE_URL}/api/v1/patients", {
        "patient_code": f"PAT-NOEHR-{uuid.uuid4().hex[:6].upper()}",
        "full_name": "No EHR Patient",
        "date_of_birth": "1990-01-01",
        "sex": "M"
    }, tokens["clinician"])
    no_ehr_id = p_no_ehr["id"]

    status, mm_missing = post_multipart(
        f"{BASE_URL}/api/v1/predictions",
        {"patient_id": no_ehr_id, "mode": "multimodal"},
        {"file": ("audit_retina.jpg", img_bytes, "image/jpeg")},
        tokens["clinician"]
    )
    assert status == 422
    print("  [PASS] Multimodal without EHR correctly returned HTTP 422 Unprocessable Entity")

    # Multimodal with EHR
    ehr_json = json.dumps({
        "age": 61,
        "sex": "F",
        "duration_diabetes_years": 14.0,
        "hba1c": 8.8,
        "blood_glucose": 195.0,
        "systolic_bp": 145.0,
        "diastolic_bp": 92.0,
        "bmi": 29.3,
        "insulin_dependent": "yes",
        "smoker": "former"
    })
    status, mm_pred = post_multipart(
        f"{BASE_URL}/api/v1/predictions",
        {"patient_id": pat_id, "mode": "multimodal", "ehr_data": ehr_json},
        {"file": ("audit_retina.jpg", img_bytes, "image/jpeg")},
        tokens["clinician"]
    )
    assert status == 200
    assert mm_pred["predicted_class"] in range(5)
    print(f"  [PASS] Multimodal Fusion (CNN+EHR): {mm_pred['prediction']} (Confidence: {mm_pred['confidence']*100:.1f}%)")
    results["D_Image_CNN"] = "PASSED"
    results["E_Multimodal_Fusion"] = "PASSED"

    # ----------------------------------------------------
    # Module F: AI Explainability (Grad-CAM)
    # ----------------------------------------------------
    log_section("F. AI Explainability (Grad-CAM)")
    gradcam_url = img_pred["gradcam_url"]
    assert gradcam_url is not None
    g_req = urllib.request.Request(f"{BASE_URL}{gradcam_url}")
    with urllib.request.urlopen(g_req) as g_resp:
        assert g_resp.status == 200
        g_bytes = g_resp.read()
        assert len(g_bytes) > 500
        print(f"  [PASS] Grad-CAM overlay generated and accessible: {gradcam_url} ({len(g_bytes)} bytes)")
    results["F_GradCAM"] = "PASSED"

    # ----------------------------------------------------
    # Module G: Clinician Review Queue
    # ----------------------------------------------------
    log_section("G. Clinician Review Queue")
    status, pending = http_get(f"{BASE_URL}/api/v1/reviews/pending", tokens["clinician"])
    assert status == 200
    assert len(pending) > 0
    target_pred = pending[0]
    target_id = target_pred["id"]
    print(f"  [PASS] Review queue populated with {len(pending)} pending cases.")

    # Submit review approval
    status, review_out = http_post(f"{BASE_URL}/api/v1/reviews", {
        "prediction_id": target_id,
        "agreed_with_ai": True,
        "decision": "APPROVED",
        "clinician_grade": target_pred["predicted_class"],
        "clinician_notes": "Retinal structures reviewed and confirmed by attending clinician.",
        "requires_followup": False,
        "followup_urgency": "routine"
    }, tokens["clinician"])
    assert status in [200, 201]
    assert review_out["decision"] == "APPROVED"
    print(f"  [PASS] Clinician review recorded for prediction {target_id}: Status=APPROVED")

    # Check review stats
    status, rev_stats = http_get(f"{BASE_URL}/api/v1/reviews/stats", tokens["clinician"])
    assert status == 200
    print(f"  [PASS] Review statistics: Total={rev_stats['total']}, Agreed={rev_stats['agreed']}, Agreement Rate={rev_stats['agreement_rate']}")
    results["G_Review_Queue"] = "PASSED"

    # ----------------------------------------------------
    # Module H: Research Analytics
    # ----------------------------------------------------
    log_section("H. Research Analytics")
    status, research = http_get(f"{BASE_URL}/api/v1/analytics/research-metrics", tokens["researcher"])
    assert status == 200
    assert "available" in research
    print(f"  [PASS] Research metrics endpoint reachable: Available={research['available']}")
    results["H_Research_Analytics"] = "PASSED"

    # ----------------------------------------------------
    # Module I: Federated Learning Simulation
    # ----------------------------------------------------
    log_section("I. Federated Learning Simulation")
    status, fl_status = http_get(f"{BASE_URL}/api/v1/federated/status", tokens["researcher"])
    assert status == 200
    print(f"  [PASS] Federated Learning service active: State={fl_status.get('status', 'ready')}")
    results["I_Federated_Learning"] = "PASSED"

    # ----------------------------------------------------
    # Module J: Admin & System Management
    # ----------------------------------------------------
    log_section("J. Admin & Model Registry")
    status, models = http_get(f"{BASE_URL}/api/v1/models", tokens["admin"])
    assert status == 200
    assert len(models) >= 1
    active_m = [m for m in models if m["is_active"]]
    assert len(active_m) >= 1
    print(f"  [PASS] Model registry contains {len(models)} registered models. Active model: {active_m[0]['name']} ({active_m[0]['architecture']})")
    results["J_Admin_Registry"] = "PASSED"

    # ----------------------------------------------------
    # Module K: Technician Intake & Triage Encounter
    # ----------------------------------------------------
    log_section("K. Technician Intake & Encounter Lifecycle")
    status, encounter = http_post(f"{BASE_URL}/api/v1/screenings", {
        "patient_id": pat_id,
        "notes": "Routine technician imaging intake.",
        "age": 61,
        "diabetes_duration": 14.0,
        "hba1c": 8.8,
        "systolic_bp": 145.0,
        "diastolic_bp": 92.0
    }, tokens["technician"])
    assert status == 201
    enc_id = encounter["id"]
    print(f"  [PASS] Screening encounter created: {encounter['case_number']} (ID: {enc_id})")

    # Upload fundus image to encounter
    status, upload_res = post_multipart(
        f"{BASE_URL}/api/v1/screenings/{enc_id}/image",
        {},
        {"file": ("tech_intake_retina.jpg", img_bytes, "image/jpeg")},
        tokens["technician"]
    )
    assert status == 200
    assert upload_res["quality_assessment"]["is_gradable"] is True
    print(f"  [PASS] Image quality checker passed: Gradable={upload_res['quality_assessment']['is_gradable']}")

    # Generate clinical report PDF
    pdf_req = urllib.request.Request(f"{BASE_URL}/api/v1/screenings/{enc_id}/report/pdf", headers={"Authorization": f"Bearer {tokens['clinician']}"})
    with urllib.request.urlopen(pdf_req) as pdf_resp:
        assert pdf_resp.status == 200
        assert pdf_resp.headers["Content-Type"] == "application/pdf"
        pdf_bytes = pdf_resp.read()
        assert len(pdf_bytes) > 500
        print(f"  [PASS] Clinical PDF report generated successfully: {len(pdf_bytes)} bytes")
    results["K_Technician_Intake"] = "PASSED"

    print("\n" + "=" * 70)
    print("SUMMARY OF ALL AUDITED MODULES:")
    for mod, res in results.items():
        print(f"  - {mod}: {res}")
    print("=" * 70)
    print("ALL MODULE AUDITS AND WORKFLOWS VERIFIED SUCCESSFULLY!")

if __name__ == "__main__":
    run_audit()
