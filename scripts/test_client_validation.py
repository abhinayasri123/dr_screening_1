import os
import sys
import io
from pathlib import Path
import numpy as np
import cv2
from PIL import Image

PROJECT_ROOT = Path(r"C:\Users\VIJAY\.gemini\antigravity\scratch\dr_screening")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.chdir(str(PROJECT_ROOT))
os.environ["SKIP_CREATE_TABLES"] = "1"

from fastapi.testclient import TestClient
from backend.main import app

client = TestClient(app)

print("=" * 80, flush=True)
print("COMPREHENSIVE RETINAL VALIDATION & PREDICTION AUDIT", flush=True)
print("=" * 80, flush=True)

# 1. Login
login_res = client.post("/api/v1/auth/login", json={"email": "admin@dr-screening.com", "password": "admin123"})
assert login_res.status_code == 200, f"Login failed: {login_res.text}"
token = login_res.json()["access_token"]
headers = {"Authorization": f"Bearer {token}"}
print("[AUTH] Logged in successfully as admin.", flush=True)

# Helper to create image byte stream
def to_bytes(arr: np.ndarray, fmt="PNG") -> io.BytesIO:
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format=fmt)
    buf.seek(0)
    return buf

# Base genuine retinal image
rf_path = list((PROJECT_ROOT / "data" / "retinal").rglob("*.png"))[0]
with Image.open(rf_path) as im:
    raw_retinal = np.array(im.convert("RGB"))

results = []

def record(test_name, expected_status, actual_status, expected_retinal, detail_or_summary):
    passed = (expected_status == actual_status)
    status_str = "PASS" if passed else "FAIL"
    print(f"[{status_str}] {test_name}: Expected HTTP {expected_status}, Got {actual_status} -> {detail_or_summary}", flush=True)
    results.append({"name": test_name, "passed": passed, "status": actual_status, "detail": detail_or_summary})

# TEST 1: Genuine Retinal Fundus (Standard)
buf = to_bytes(raw_retinal)
r1 = client.post("/api/v1/predictions/", headers=headers, files={"file": ("raw_fundus.png", buf, "image/png")})
j1 = r1.json() if r1.status_code == 200 else {}
record("1. Genuine Retinal (Standard)", 200, r1.status_code, True, f"Class={j1.get('predicted_label')} | Status={j1.get('summary_status')}")

# TEST 2: Genuine Retinal with Center Square Crop (No Circular Border)
h, w = raw_retinal.shape[:2]
# Resize raw retinal to 400x400 first so crop is 200x200
big_ret = cv2.resize(raw_retinal, (400, 400))
center_crop = big_ret[100:300, 100:300]
buf = to_bytes(center_crop)
r2 = client.post("/api/v1/predictions/", headers=headers, files={"file": ("square_crop_fundus.png", buf, "image/png")})
j2 = r2.json() if r2.status_code == 200 else {}
record("2. Genuine Retinal (Square Crop, No Circle)", 200, r2.status_code, True, f"Class={j2.get('predicted_label')} | Status={j2.get('summary_status')}")

# TEST 3: Genuine Retinal with Cool Timestamp / Watermark Overlay
ts_img = raw_retinal.copy()
cv2.rectangle(ts_img, (5, 5), (60, 25), (0, 200, 255), -1) # Cyan camera box
buf = to_bytes(ts_img)
r3 = client.post("/api/v1/predictions/", headers=headers, files={"file": ("timestamp_fundus.png", buf, "image/png")})
j3 = r3.json() if r3.status_code == 200 else {}
record("3. Genuine Retinal (With Timestamp Overlay)", 200, r3.status_code, True, f"Class={j3.get('predicted_label')} | Status={j3.get('summary_status')}")

# TEST 4: Genuine Retinal with Severe Exudates / Glare
ex_img = raw_retinal.copy()
for cx, cy in [(100, 100), (120, 110), (110, 130), (90, 120), (130, 95)]:
    cv2.circle(ex_img, (cx, cy), 14, (240, 240, 240), -1)
buf = to_bytes(ex_img)
r4 = client.post("/api/v1/predictions/", headers=headers, files={"file": ("exudates_fundus.png", buf, "image/png")})
j4 = r4.json() if r4.status_code == 200 else {}
record("4. Genuine Retinal (Severe DR Exudates)", 200, r4.status_code, True, f"Class={j4.get('predicted_label')} | Status={j4.get('summary_status')}")

# TEST 5: Blurry Retinal Image (Borderline Optical Quality -> INCONCLUSIVE, NOT 422!)
blurred_ret = cv2.GaussianBlur(raw_retinal, (51, 51), 30)
buf = to_bytes(blurred_ret)
r5 = client.post("/api/v1/predictions/", headers=headers, files={"file": ("blurry_fundus.png", buf, "image/png")})
j5 = r5.json() if r5.status_code == 200 else {}
is_inconclusive = (j5.get("summary_status") == "INCONCLUSIVE — CLINICIAN REVIEW REQUIRED")
record("5. Blurry Retinal Image", 200, r5.status_code, True, f"SummaryStatus={j5.get('summary_status')} (Inconclusive={is_inconclusive})")

# TEST 6: Non-Retinal - Human Portrait / Selfie
bench_dir = PROJECT_ROOT / "test_benchmark_dataset"
p_path = bench_dir / "portrait_selfie_1.png"
with open(p_path, "rb") as f:
    r6 = client.post("/api/v1/predictions/", headers=headers, files={"file": ("portrait.png", f, "image/png")})
record("6. Non-Retinal (Human Portrait)", 422, r6.status_code, False, f"Detail={r6.json().get('detail', '')[:60]}...")

# TEST 7: Non-Retinal - Ordinary Photograph (Nature / Room)
o_path = bench_dir / "ordinary_photo_1.png"
with open(o_path, "rb") as f:
    r7 = client.post("/api/v1/predictions/", headers=headers, files={"file": ("nature.png", f, "image/png")})
record("7. Non-Retinal (Ordinary Photo)", 422, r7.status_code, False, f"Detail={r7.json().get('detail', '')[:60]}...")

# TEST 8: Non-Retinal - Document / Screenshot
d_path = bench_dir / "document_screenshot_1.png"
with open(d_path, "rb") as f:
    r8 = client.post("/api/v1/predictions/", headers=headers, files={"file": ("doc.png", f, "image/png")})
record("8. Non-Retinal (Document/Screenshot)", 422, r8.status_code, False, f"Detail={r8.json().get('detail', '')[:60]}...")

# TEST 9: Non-Retinal - Animal Photograph
a_path = bench_dir / "animal_photo_1.png"
with open(a_path, "rb") as f:
    r9 = client.post("/api/v1/predictions/", headers=headers, files={"file": ("animal.png", f, "image/png")})
record("9. Non-Retinal (Animal Photo)", 422, r9.status_code, False, f"Detail={r9.json().get('detail', '')[:60]}...")

# TEST 10: Non-Retinal - Grayscale Radiograph (Chest X-ray)
x_path = bench_dir / "chest_xray_1.png"
with open(x_path, "rb") as f:
    r10 = client.post("/api/v1/predictions/", headers=headers, files={"file": ("xray.png", f, "image/png")})
record("10. Non-Retinal (Chest X-ray)", 422, r10.status_code, False, f"Detail={r10.json().get('detail', '')[:60]}...")

# TEST 11: Non-Retinal - Cartoon Artwork
c_path = bench_dir / "cartoon_artwork_1.png"
with open(c_path, "rb") as f:
    r11 = client.post("/api/v1/predictions/", headers=headers, files={"file": ("cartoon.png", f, "image/png")})
record("11. Non-Retinal (Cartoon Artwork)", 422, r11.status_code, False, f"Detail={r11.json().get('detail', '')[:60]}...")

# TEST 12: Corrupted File
corrupt_buf = io.BytesIO(b"NOT_A_VALID_IMAGE_HEADER_BYTES_123456789")
r12 = client.post("/api/v1/predictions/", headers=headers, files={"file": ("corrupt.jpg", corrupt_buf, "image/jpeg")})
record("12. Corrupted / Unreadable File", 422, r12.status_code, False, f"Detail={r12.json().get('detail', '')[:60]}...")

# TEST 13: Screening Encounter Workflow (Create -> Upload Blurry Image -> Predict)
# 1. Create Patient & Encounter
pat_res = client.post("/api/v1/patients/", headers=headers, json={
    "full_name": "Audit Test Patient",
    "patient_code": "PAT-AUDIT-999",
    "date_of_birth": "1970-01-01",
    "gender": "Other"
})
pat_id = pat_res.json()["id"] if pat_res.status_code in [200, 201] else client.get("/api/v1/patients/", headers=headers).json()[0]["id"]

enc_res = client.post("/api/v1/screenings/", headers=headers, json={"patient_id": pat_id, "notes": "Validation audit test"})
assert enc_res.status_code == 201, f"Encounter creation failed: {enc_res.text}"
enc_id = enc_res.json()["id"]

# 2. Upload Blurry Retinal Image
buf_b = to_bytes(blurred_ret)
up_res = client.post(f"/api/v1/screenings/{enc_id}/image", headers=headers, files={"file": ("blurry_enc.png", buf_b, "image/png")})
assert up_res.status_code == 200, f"Upload failed: {up_res.text}"

# 3. Run Prediction on Blurry Encounter (MUST SUCCEED HTTP 200 with INCONCLUSIVE, NOT FAIL WITH 422!)
pred_enc_res = client.post(f"/api/v1/screenings/{enc_id}/predict", headers=headers)
j_enc = pred_enc_res.json() if pred_enc_res.status_code == 200 else {}
record("13. Screening Encounter Workflow (Blurry Retinal Prediction)", 200, pred_enc_res.status_code, True, f"Status={j_enc.get('summary_status')}")

# Print Total Summary
n_passed = sum(1 for r in results if r["passed"])
print("=" * 80, flush=True)
print(f"AUDIT SUMMARY: {n_passed} / {len(results)} Tests Passed ({n_passed/len(results)*100:.1f}%)", flush=True)
print("=" * 80, flush=True)

assert n_passed == len(results), f"Some tests failed! {n_passed}/{len(results)}"
