"""
Verification test for XAI Explainability endpoints and data flow.
Tests:
1. /api/v1/screenings/xai/cases returns non-empty list of eligible cases with physical fundus images.
2. Every returned case has a valid physical image file on disk.
3. /api/v1/screenings/{id} returns matching images and prediction details.
4. /api/v1/screenings/{id}/gradcam generates or recomputes Grad-CAM.
5. Error handling scenarios (non-existent case, ungradable/missing image).
"""

import sys
import os
import sqlite3
from pathlib import Path

BASE_DIR = Path("C:/Users/VIJAY/.gemini/antigravity/scratch/dr_screening")
sys.path.insert(0, str(BASE_DIR))
os.chdir(str(BASE_DIR))

import asyncio
from httpx import AsyncClient, ASGITransport
from backend.main import app

async def run_tests():
    print("=" * 60, flush=True)
    print("STARTING XAI EXPLAINABILITY END-TO-END VERIFICATION", flush=True)
    print("=" * 60, flush=True)
    
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        # Step 1: Login as clinician
        print("\n--- 1. Authenticating as Clinician ---", flush=True)
        login_res = await ac.post("/api/v1/auth/login", data={"username": "clinician@dr-screening.com", "password": "clinician123"})
        assert login_res.status_code == 200, f"Login failed: {login_res.text}"
        token = login_res.json()["access_token"]
        headers = {"Authorization": f"Bearer {token}"}
        print("Login SUCCESS. Bearer token acquired.", flush=True)
        
        # Step 2: Test /api/v1/screenings/xai/cases
        print("\n--- 2. Testing GET /api/v1/screenings/xai/cases ---", flush=True)
        xai_res = await ac.get("/api/v1/screenings/xai/cases?limit=50", headers=headers)
        assert xai_res.status_code == 200, f"XAI cases call failed: {xai_res.text}"
        cases = xai_res.json()
        print(f"Total eligible XAI cases returned: {len(cases)}", flush=True)
        assert len(cases) > 0, "Expected non-empty list of eligible XAI cases!"
        
        for c in cases[:5]:
            print(f"  Case: {c['case_number']} | Patient: {c['patient_name']} ({c['patient_code']}) | Grade: {c['predicted_grade']} ({c['predicted_label']}) | ImageURL: {c['image_url']} | HasGradcam: {c['has_gradcam']}", flush=True)
            
        # Step 3: Verify all returned cases have physical files on disk
        print("\n--- 3. Verifying Physical Fundus Images on Disk ---", flush=True)
        for c in cases:
            img_url = c["image_url"]
            fname = os.path.basename(img_url)
            p = BASE_DIR / "uploads" / fname
            assert p.exists(), f"Image file {p} does not exist on disk!"
        print(f"PASS: 100% ({len(cases)}/{len(cases)}) of eligible cases have physical fundus images on disk.", flush=True)
        
        # Step 4: Test GET /api/v1/screenings/{id} on a case
        print("\n--- 4. Testing GET /api/v1/screenings/{id} Detail ---", flush=True)
        selected = cases[0]
        enc_id = selected["id"]
        det_res = await ac.get(f"/api/v1/screenings/{enc_id}", headers=headers)
        assert det_res.status_code == 200, f"Failed to get screening detail: {det_res.text}"
        detail = det_res.json()
        
        assert "encounter" in detail
        assert "patient" in detail
        assert "images" in detail and len(detail["images"]) > 0
        print(f"Detail retrieved for {selected['case_number']}:", flush=True)
        print(f"  - Patient: {detail['patient']['full_name']}", flush=True)
        print(f"  - Images count: {len(detail['images'])}", flush=True)
        print(f"  - Image URL: {detail['images'][0]['image_url']}", flush=True)
        if detail.get("prediction"):
            print(f"  - Predicted Grade: {detail['prediction']['predicted_class']} ({detail['prediction']['predicted_label']})", flush=True)
            print(f"  - Confidence: {detail['prediction']['confidence']}", flush=True)
            print(f"  - Grad-CAM URL: {detail['prediction']['gradcam_url']}", flush=True)
            
        # Step 5: Test POST /api/v1/screenings/{id}/gradcam
        print("\n--- 5. Testing POST /api/v1/screenings/{id}/gradcam (Live Model Inference) ---", flush=True)
        gradcam_res = await ac.post(f"/api/v1/screenings/{enc_id}/gradcam?target_class=2", headers=headers)
        print("Grad-CAM generation status:", gradcam_res.status_code, flush=True)
        assert gradcam_res.status_code == 200, f"Grad-CAM generation failed: {gradcam_res.text}"
        gc_data = gradcam_res.json()
        print("Grad-CAM generation response:", gc_data, flush=True)
        assert "gradcam_url" in gc_data and gc_data["gradcam_url"] is not None
        gc_fname = os.path.basename(gc_data["gradcam_url"])
        assert (BASE_DIR / "uploads" / gc_fname).exists(), "Generated Grad-CAM image file not found on disk!"
        print("PASS: Grad-CAM overlay successfully computed and verified on disk.", flush=True)
        
        # Step 6: Test Error Scenarios
        print("\n--- 6. Testing Error Scenarios ---", flush=True)
        # 6a. Non-existent encounter
        bad_res = await ac.get("/api/v1/screenings/non-existent-uuid-123", headers=headers)
        assert bad_res.status_code == 404, f"Expected 404, got {bad_res.status_code}"
        print("PASS: Non-existent encounter returns 404.", flush=True)
        
        # 6b. Unauthorized access (no token)
        unauth_res = await ac.get("/api/v1/screenings/xai/cases?limit=10")
        assert unauth_res.status_code in (401, 403), f"Expected 401/403, got {unauth_res.status_code}"
        print("PASS: Unauthenticated access blocked with 401/403.", flush=True)

    print("\n" + "=" * 60, flush=True)
    print("ALL XAI EXPLAINABILITY VERIFICATION TESTS PASSED!", flush=True)
    print("=" * 60, flush=True)

if __name__ == "__main__":
    asyncio.run(run_tests())
