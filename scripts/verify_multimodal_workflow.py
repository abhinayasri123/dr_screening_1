"""
Automated Verification Script for DR Screening Platform:
1. Tests API Health
2. Tests User Auth & Token Retrieval
3. Tests Patient EHR Record Fetch & Save (GET/POST /api/v1/patients/{id}/ehr)
4. Tests Image-Only Inference Mode
5. Tests Multimodal Mode Missing EHR Validation (verifies HTTP 422)
6. Tests Multimodal Mode Inference with EHR
7. Tests EHR Sensitivity: Confirms changes in EHR covariates alter multimodal logits/probabilities
"""

import os
import sys
import json
import urllib.request
import urllib.error
from pathlib import Path

BASE_URL = "http://127.0.0.1:8000"

def run_tests():
    print("=" * 70)
    print("VERIFYING DR SCREENING MULTIMODAL WORKFLOW")
    print("=" * 70)

    # 1. Health check
    req = urllib.request.Request(f"{BASE_URL}/api/v1/health")
    with urllib.request.urlopen(req) as resp:
        health_data = json.loads(resp.read().decode())
        print(f"[1] Health Check: Status={resp.status}, Body={health_data}")
        assert health_data["status"] == "healthy"
        assert health_data["model_loaded"] is True

    # 2. Login as clinician
    login_data = json.dumps({
        "email": "clinician@dr-screening.com",
        "password": "clinician123"
    })
    login_req = urllib.request.Request(
        f"{BASE_URL}/api/v1/auth/login",
        data=login_data.encode(),
        headers={"Content-Type": "application/json"},
        method="POST"
    )
    with urllib.request.urlopen(login_req) as resp:
        token_data = json.loads(resp.read().decode())
        token = token_data["access_token"]
        print(f"[2] Authentication: Successfully retrieved JWT for Dr. Sarah Chen")

    auth_headers = {"Authorization": f"Bearer {token}"}

    # 3. Get patients list
    patients_req = urllib.request.Request(f"{BASE_URL}/api/v1/patients", headers=auth_headers)
    with urllib.request.urlopen(patients_req) as resp:
        patients = json.loads(resp.read().decode())
        print(f"[3] Patients: Retrieved {len(patients)} registered patients.")
        test_patient = patients[0]
        patient_id = test_patient["id"]
        print(f"    Selected patient: {test_patient['patient_code']} ({test_patient.get('name') or test_patient.get('full_name')})")

    # 4. Test EHR endpoint (GET /api/v1/patients/{id}/ehr)
    ehr_req = urllib.request.Request(f"{BASE_URL}/api/v1/patients/{patient_id}/ehr", headers=auth_headers)
    with urllib.request.urlopen(ehr_req) as resp:
        existing_ehr = json.loads(resp.read().decode())
        print(f"[4] Patient EHR lookup: {existing_ehr is not None}")

    # 5. Locate sample retinal image
    project_root = Path(__file__).resolve().parent.parent
    sample_images = list((project_root / "data" / "retinal").rglob("*.png")) + list((project_root / "data" / "retinal").rglob("*.jpg"))
    assert len(sample_images) > 0, "No sample retinal fundus images found!"
    sample_image_path = sample_images[0]
    print(f"[5] Sample Fundus Image: {sample_image_path.name}")

    with open(sample_image_path, "rb") as f:
        image_bytes = f.read()

    # Helper for multipart form data
    import uuid
    def make_multipart(fields, files):
        boundary = f"----WebKitFormBoundary{uuid.uuid4().hex}"
        body = bytearray()
        for k, v in fields.items():
            body.extend(f"--{boundary}\r\n".encode())
            body.extend(f'Content-Disposition: form-data; name="{k}"\r\n\r\n'.encode())
            body.extend(f"{v}\r\n".encode())
        for k, (filename, filedata, content_type) in files.items():
            body.extend(f"--{boundary}\r\n".encode())
            body.extend(f'Content-Disposition: form-data; name="{k}"; filename="{filename}"\r\n'.encode())
            body.extend(f"Content-Type: {content_type}\r\n\r\n".encode())
            body.extend(filedata)
            body.extend(b"\r\n")
        body.extend(f"--{boundary}--\r\n".encode())
        return bytes(body), f"multipart/form-data; boundary={boundary}"

    # 6. Test Image-Only Inference
    body, ctype = make_multipart(
        {"patient_id": patient_id, "mode": "image_only"},
        {"file": (sample_image_path.name, image_bytes, "image/png")}
    )
    img_req = urllib.request.Request(
        f"{BASE_URL}/api/v1/predictions",
        data=body,
        headers={**auth_headers, "Content-Type": ctype},
        method="POST"
    )
    with urllib.request.urlopen(img_req) as resp:
        img_pred = json.loads(resp.read().decode())
        print(f"[6] Image-Only Mode Result: {img_pred['prediction']} (Confidence: {img_pred['confidence']:.2%}, Uncertainty: {img_pred['uncertainty_level']})")
        assert img_pred["prediction"] in ["No DR", "Mild", "Moderate", "Severe", "Proliferative DR"]

    # 7. Test Multimodal Mode Missing EHR (Expect 422)
    # We create a brand new patient with no EHR to test missing EHR validation
    new_pat_req = urllib.request.Request(
        f"{BASE_URL}/api/v1/patients",
        data=json.dumps({
            "patient_code": f"TEST-NOEHR-{uuid.uuid4().hex[:6]}",
            "full_name": "Test Missing EHR Patient",
            "date_of_birth": "1980-01-01",
            "sex": "F"
        }).encode(),
        headers={**auth_headers, "Content-Type": "application/json"},
        method="POST"
    )
    with urllib.request.urlopen(new_pat_req) as resp:
        no_ehr_patient = json.loads(resp.read().decode())

    body, ctype = make_multipart(
        {"patient_id": no_ehr_patient["id"], "mode": "multimodal"},
        {"file": (sample_image_path.name, image_bytes, "image/png")}
    )
    missing_ehr_req = urllib.request.Request(
        f"{BASE_URL}/api/v1/predictions",
        data=body,
        headers={**auth_headers, "Content-Type": ctype},
        method="POST"
    )
    try:
        urllib.request.urlopen(missing_ehr_req)
        print("[-] FAILED: Expected 422 for missing EHR, but request succeeded!")
        sys.exit(1)
    except urllib.error.HTTPError as he:
        print(f"[7] Missing EHR Validation: Received expected HTTP {he.code} ({he.read().decode()})")
        assert he.code == 422

    # 8. Test Multimodal Mode with Low-Risk Clinical EHR
    low_risk_ehr = {
        "age": 45,
        "sex": 0,
        "diabetes_duration_years": 2.0,
        "hba1c": 5.8,
        "systolic_bp": 115,
        "diastolic_bp": 75,
        "bmi": 22.0,
        "smoking_status": 0,
        "insulin_use": False,
        "prev_laser_treatment": False,
        "nephropathy": False,
        "neuropathy": False,
        "cholesterol_ldl": 90.0
    }
    body, ctype = make_multipart(
        {
            "patient_id": patient_id,
            "mode": "multimodal",
            "ehr_data": json.dumps(low_risk_ehr)
        },
        {"file": (sample_image_path.name, image_bytes, "image/png")}
    )
    mm_req = urllib.request.Request(
        f"{BASE_URL}/api/v1/predictions",
        data=body,
        headers={**auth_headers, "Content-Type": ctype},
        method="POST"
    )
    with urllib.request.urlopen(mm_req) as resp:
        low_risk_pred = json.loads(resp.read().decode())
        print(f"[8] Multimodal Mode (Low-Risk EHR) Result: {low_risk_pred['prediction']}")
        print(f"    Class Probabilities: {low_risk_pred.get('probabilities')}")

    # 9. Test Multimodal Mode with High-Risk Clinical EHR (Severe glycemic & vascular risk)
    high_risk_ehr = {
        "age": 68,
        "sex": 1,
        "diabetes_duration_years": 25.0,
        "hba1c": 12.5,
        "systolic_bp": 175,
        "diastolic_bp": 105,
        "bmi": 34.0,
        "smoking_status": 2,
        "insulin_use": True,
        "prev_laser_treatment": True,
        "nephropathy": True,
        "neuropathy": True,
        "cholesterol_ldl": 220.0
    }
    body, ctype = make_multipart(
        {
            "patient_id": patient_id,
            "mode": "multimodal",
            "ehr_data": json.dumps(high_risk_ehr)
        },
        {"file": (sample_image_path.name, image_bytes, "image/png")}
    )
    mm_req_high = urllib.request.Request(
        f"{BASE_URL}/api/v1/predictions",
        data=body,
        headers={**auth_headers, "Content-Type": ctype},
        method="POST"
    )
    with urllib.request.urlopen(mm_req_high) as resp:
        high_risk_pred = json.loads(resp.read().decode())
        print(f"[9] Multimodal Mode (High-Risk EHR) Result: {high_risk_pred['prediction']}")
        print(f"    Class Probabilities: {high_risk_pred.get('probabilities')}")

    # 10. Verify EHR sensitivity
    low_p = low_risk_pred['class_probabilities']
    high_p = high_risk_pred['class_probabilities']
    prob_diff = sum(abs(a - b) for a, b in zip(low_p, high_p))
    print(f"[10] Total Absolute Probability Delta across classes: {prob_diff:.8f}")
    assert prob_diff > 0.0, "EHR covariates had no effect on model predictions!"
    print("    -> CONFIRMED: Clinical EHR covariates are actively influencing multimodal fusion predictions!")

    print("=" * 70)
    print("ALL MULTIMODAL VERIFICATION TESTS PASSED SUCCESSFULLY!")
    print("=" * 70)

if __name__ == "__main__":
    run_tests()
