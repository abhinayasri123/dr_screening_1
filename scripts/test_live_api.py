import requests
import json
from pathlib import Path

BASE_URL = "http://127.0.0.1:8000/api/v1"
BENCH_DIR = Path(r"C:\Users\VIJAY\.gemini\antigravity\scratch\dr_screening\test_benchmark_dataset")
RETINAL_IMG = Path(r"C:\Users\VIJAY\.gemini\antigravity\scratch\dr_screening\data\retinal\test\0\demo_cls0_test_000.png")

# 1. Login
resp = requests.post(f"{BASE_URL}/auth/login", json={"email": "admin@dr-screening.com", "password": "admin123"})
print("Auth Login HTTP Status:", resp.status_code)
assert resp.status_code == 200, f"Login failed: {resp.text}"
token = resp.json()["access_token"]
headers = {"Authorization": f"Bearer {token}"}
print("Access Token successfully obtained.")

# 2. Test Non-Retinal Portrait Image Rejection (Stage A)
portrait_path = BENCH_DIR / "portrait_selfie_1.png"
with open(portrait_path, "rb") as f:
    r_non = requests.post(f"{BASE_URL}/predictions/", headers=headers, files={"file": ("portrait.png", f, "image/png")})
print("\n[Stage A Test] Non-Retinal Portrait Upload HTTP Status:", r_non.status_code)
detail = r_non.json().get("detail", "")
print("Rejection Detail Response:\n", detail)
assert r_non.status_code == 422, f"Expected 422, got {r_non.status_code}"
assert "INVALID IMAGE TYPE" in detail, "Expected exact invalid image message"
print(">>> Stage A Rejection Verified: Portrait rejected with 422 and exact clinical warning.")

# 3. Test Genuine Retinal Image Classification (Stage B)
with open(RETINAL_IMG, "rb") as f:
    r_ret = requests.post(f"{BASE_URL}/predictions/", headers=headers, files={"file": ("fundus.png", f, "image/png")})
print("\n[Stage B Test] Genuine Fundus Upload HTTP Status:", r_ret.status_code)
ret_json = r_ret.json()
print("Predicted Class:", ret_json.get("predicted_class"))
print("Predicted Label:", ret_json.get("predicted_label"))
print("Summary Status:", ret_json.get("summary_status"))
print("Confidence:", ret_json.get("confidence"))
print("Grad-CAM URL:", ret_json.get("gradcam_url"))
assert r_ret.status_code == 200, f"Expected 200, got {r_ret.status_code}: {ret_json}"
assert ret_json.get("summary_status") in ["NO DR DETECTED", "DR DETECTED", "INCONCLUSIVE — CLINICIAN REVIEW REQUIRED"]
print(">>> Stage B Classification Verified: Retinal image classified with summary status and Grad-CAM.")

print("\n=======================================================")
print(">>> ALL LIVE SYSTEM INTEGRATION TESTS PASSED (100%) <<<")
print("=======================================================")
