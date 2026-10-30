"""
End-to-End TestClient Verification for XAI Explainability Workflow.
Tests:
1. Clinician Authentication & Token Generation
2. GET /api/v1/screenings/xai/cases
3. Physical existence of all retinal image files on disk
4. GET /api/v1/screenings/{id} details (encounter, patient, images, prediction)
5. POST /api/v1/screenings/{id}/gradcam (Grad-CAM generation with EfficientNet-B3)
6. Error handling (non-existent encounter returns 404, missing token returns 401)
"""

import sys
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fastapi.testclient import TestClient
from backend.main import app

def run_xai_verification():
    print("=" * 65, flush=True)
    print("STARTING XAI EXPLAINABILITY END-TO-END VERIFICATION", flush=True)
    print("=" * 65, flush=True)

    with TestClient(app) as client:
        # Step 1: Login
        print("\n--- 1. Authenticating as Clinician ---", flush=True)
        login_res = client.post(
            "/api/v1/auth/login",
            json={"email": "clinician@dr-screening.com", "password": "clinician123"}
        )
        print(f"Login HTTP Status: {login_res.status_code}", flush=True)
        assert login_res.status_code == 200, f"Login failed: {login_res.text}"
        token = login_res.json()["access_token"]
        headers = {"Authorization": f"Bearer {token}"}
        print("PASS: Clinician JWT token acquired successfully.", flush=True)

        # Step 2: XAI Cases Endpoint
        print("\n--- 2. Testing GET /api/v1/screenings/xai/cases ---", flush=True)
        xai_res = client.get("/api/v1/screenings/xai/cases?limit=50", headers=headers)
        print(f"XAI Cases HTTP Status: {xai_res.status_code}", flush=True)
        assert xai_res.status_code == 200, f"Failed to get XAI cases: {xai_res.text}"
        cases = xai_res.json()
        print(f"PASS: Returned {len(cases)} eligible XAI screening cases.", flush=True)
        assert len(cases) > 0, "No XAI cases returned!"

        print("\nSample XAI cases returned for dropdown:")
        for i, c in enumerate(cases[:5]):
            print(f"  [{i+1}] Case: {c['case_number']} | Patient: {c['patient_name']} ({c['patient_code']}) | Grade: {c['predicted_grade']} ({c['predicted_label']}) | HasGradcam: {c['has_gradcam']}", flush=True)

        # Step 3: Physical Image File Verification
        print("\n--- 3. Verifying Physical Fundus Images on Disk ---", flush=True)
        missing_count = 0
        for c in cases:
            img_url = c["image_url"]
            fname = os.path.basename(img_url)
            p = ROOT / "uploads" / fname
            if not p.exists():
                print(f"  MISSING: {p}", flush=True)
                missing_count += 1
        assert missing_count == 0, f"Found {missing_count} missing image files!"
        print(f"PASS: 100% ({len(cases)}/{len(cases)}) of cases have physical fundus images on disk.", flush=True)

        # Step 4: Detail Endpoint for Selected Case
        print("\n--- 4. Testing GET /api/v1/screenings/{id} Detail ---", flush=True)
        selected = cases[0]
        enc_id = selected["id"]
        det_res = client.get(f"/api/v1/screenings/{enc_id}", headers=headers)
        print(f"Detail HTTP Status: {det_res.status_code}", flush=True)
        assert det_res.status_code == 200, f"Detail call failed: {det_res.text}"
        detail = det_res.json()
        
        assert "encounter" in detail
        assert "patient" in detail
        assert "images" in detail and len(detail["images"]) > 0
        img0 = detail["images"][0]
        print(f"Selected Encounter: {detail['encounter']['case_number']}", flush=True)
        print(f"  - Patient: {detail['patient']['full_name']} ({detail['patient']['patient_code']})", flush=True)
        print(f"  - Fundus Image URL: {img0['image_url']}", flush=True)
        print(f"  - Blur Score (Laplacian): {img0['blur_score']} | Gradable: {img0['is_gradable']}", flush=True)
        if detail.get("prediction"):
            pred = detail["prediction"]
            print(f"  - Predicted Grade: Class {pred['predicted_class']} ({pred['predicted_label']})", flush=True)
            print(f"  - Softmax Confidence: {pred['confidence'] * 100:.1f}%", flush=True)
            print(f"  - MC Dropout Uncertainty: {pred['uncertainty_level']} (std = {pred['uncertainty_std']:.4f})", flush=True)
            print(f"  - Grad-CAM Heatmap URL: {pred['gradcam_url']}", flush=True)
        print("PASS: Detail record contains coherent encounter, patient, image, and prediction data.", flush=True)

        # Step 5: Live Grad-CAM Generation / Recomputation
        print("\n--- 5. Testing POST /api/v1/screenings/{id}/gradcam ---", flush=True)
        gc_res = client.post(f"/api/v1/screenings/{enc_id}/gradcam?target_class=2", headers=headers)
        print(f"Grad-CAM Generation HTTP Status: {gc_res.status_code}", flush=True)
        assert gc_res.status_code == 200, f"Grad-CAM generation failed: {gc_res.text}"
        gc_data = gc_res.json()
        print(f"Grad-CAM Response: Status={gc_data.get('status')}, URL={gc_data.get('gradcam_url')}", flush=True)
        assert "gradcam_url" in gc_data and gc_data["gradcam_url"] is not None
        gc_fname = os.path.basename(gc_data["gradcam_url"])
        gc_disk_path = ROOT / "uploads" / gc_fname
        assert gc_disk_path.exists(), f"Grad-CAM file {gc_disk_path} not found on disk!"
        print(f"PASS: Grad-CAM overlay generated ({gc_disk_path.stat().st_size} bytes) and physically verified on disk.", flush=True)

        # Step 6: Negative & Error Scenarios
        print("\n--- 6. Testing Error Scenarios ---", flush=True)
        bad_res = client.get("/api/v1/screenings/non-existent-screening-id-999", headers=headers)
        assert bad_res.status_code == 404, f"Expected 404 for missing ID, got {bad_res.status_code}"
        print(f"PASS: Non-existent screening encounter returns 404 Not Found ({bad_res.json()}).", flush=True)

        unauth_res = client.get("/api/v1/screenings/xai/cases?limit=10")
        assert unauth_res.status_code in (401, 403), f"Expected 401/403 for unauthorized call, got {unauth_res.status_code}"
        print(f"PASS: Unauthenticated call correctly blocked with {unauth_res.status_code} ({unauth_res.json()}).", flush=True)

    print("\n" + "=" * 65, flush=True)
    print("ALL 6 XAI EXPLAINABILITY VERIFICATION TESTS PASSED SUCCESSFULLY!", flush=True)
    print("=" * 65, flush=True)

if __name__ == "__main__":
    run_xai_verification()
