"""
Live End-to-End Verification of XAI Explainability Workflow against http://127.0.0.1:8000
"""

import urllib.request
import urllib.parse
import json
import os
from pathlib import Path

BASE_DIR = Path("C:/Users/VIJAY/.gemini/antigravity/scratch/dr_screening")
API_HOST = "http://127.0.0.1:8000"

def post_json(url, data, headers=None):
    headers = headers or {}
    headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=json.dumps(data).encode("utf-8"), headers=headers, method="POST")
    with urllib.request.urlopen(req) as resp:
        return resp.status, json.loads(resp.read().decode("utf-8"))

def get_json(url, headers=None):
    headers = headers or {}
    req = urllib.request.Request(url, headers=headers, method="GET")
    with urllib.request.urlopen(req) as resp:
        return resp.status, json.loads(resp.read().decode("utf-8"))

def test_live():
    print("=" * 65)
    print("TESTING LIVE SERVER AT http://127.0.0.1:8000")
    print("=" * 65)

    # 1. Clinician Login
    print("\n1. Logging in as Clinician...")
    status, auth_data = post_json(f"{API_HOST}/api/v1/auth/login", {
        "email": "clinician@dr-screening.com",
        "password": "clinician123"
    })
    print(f"Login Status: {status}")
    assert status == 200, f"Login failed: {auth_data}"
    token = auth_data["access_token"]
    headers = {"Authorization": f"Bearer {token}"}
    print("PASS: Access token acquired.")

    # 2. Query /api/v1/screenings/xai/cases
    print("\n2. Querying /api/v1/screenings/xai/cases...")
    status, cases = get_json(f"{API_HOST}/api/v1/screenings/xai/cases?limit=50", headers=headers)
    print(f"XAI Cases Status: {status}, Total Eligible Cases: {len(cases)}")
    assert status == 200 and len(cases) > 0, "No XAI cases returned!"

    print("First 5 XAI dropdown options:")
    for i, c in enumerate(cases[:5]):
        print(f"  [{i+1}] {c['case_number']} | Patient: {c['patient_name']} ({c['patient_code']}) | Grade: {c['predicted_grade']} ({c['predicted_label']}) | Gradable: {c['has_image']} | HasGradcam: {c['has_gradcam']}")

    # 3. Verify images on disk
    print("\n3. Verifying physical existence of all fundus images on disk...")
    for c in cases:
        fname = os.path.basename(c["image_url"])
        full_p = BASE_DIR / "uploads" / fname
        assert full_p.exists(), f"Image file not found: {full_p}"
    print(f"PASS: 100% of images ({len(cases)} files) physically exist on disk.")

    # 4. Query screening details for first case
    selected_id = cases[0]["id"]
    print(f"\n4. Querying full screening detail for {cases[0]['case_number']} ({selected_id})...")
    status, detail = get_json(f"{API_HOST}/api/v1/screenings/{selected_id}", headers=headers)
    assert status == 200, f"Detail failed: {detail}"
    print(f"Encounter: {detail['encounter']['case_number']}")
    print(f"Patient: {detail['patient']['full_name']}")
    print(f"Image URL: {detail['images'][0]['image_url']}")
    if detail.get("prediction"):
        p = detail["prediction"]
        print(f"Prediction: Grade {p['predicted_class']} ({p['predicted_label']}) | Confidence: {p['confidence']*100:.1f}%")
        print(f"Grad-CAM Heatmap URL: {p['gradcam_url']}")

    # 5. Live Grad-CAM computation
    print(f"\n5. Computing / Recomputing Grad-CAM for {selected_id}...")
    status, gc_res = post_json(f"{API_HOST}/api/v1/screenings/{selected_id}/gradcam?target_class=2", {}, headers=headers)
    assert status == 200, f"Grad-CAM generation failed: {gc_res}"
    print(f"Grad-CAM Response: Status={gc_res.get('status')}, URL={gc_res.get('gradcam_url')}")
    gc_fname = os.path.basename(gc_res["gradcam_url"])
    gc_file = BASE_DIR / "uploads" / gc_fname
    assert gc_file.exists(), f"Grad-CAM file {gc_file} not found on disk!"
    print(f"PASS: Grad-CAM overlay generated ({gc_file.stat().st_size} bytes) and verified on disk.")

    # 6. Verify fallback /screenings?limit=50&has_image=true
    print("\n6. Testing /api/v1/screenings?limit=50&has_image=true...")
    status, sc_cases = get_json(f"{API_HOST}/api/v1/screenings?limit=50&has_image=true", headers=headers)
    assert status == 200, f"Fallback screenings failed: {sc_cases}"
    print(f"Returned {len(sc_cases)} cases with verified images.")
    assert len(sc_cases) > 0

    print("\n" + "=" * 65)
    print("ALL LIVE END-TO-END VERIFICATION CHECKS PASSED SUCCESSFULLY!")
    print("=" * 65)

if __name__ == "__main__":
    test_live()
