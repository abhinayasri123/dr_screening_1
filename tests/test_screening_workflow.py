import io
import os
import pytest
import asyncio
from functools import wraps
from PIL import Image
import httpx

from backend.main import app

def async_test(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        return asyncio.run(f(*args, **kwargs))
    return wrapper

@async_test
async def test_full_clinician_in_the_loop_workflow():
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        # Step 0: Authentication
        print("\n--- STEP 0: Authentication ---", flush=True)
        adm_login = await client.post("/api/v1/auth/login", json={"email": "admin@dr-screening.com", "password": "admin123"})
        assert adm_login.status_code == 200, f"Admin login failed: {adm_login.text}"
        admin_headers = {"Authorization": f"Bearer {adm_login.json()['access_token']}"}

        clin_login = await client.post("/api/v1/auth/login", json={"email": "clinician@dr-screening.com", "password": "clinician123"})
        assert clin_login.status_code == 200, f"Clinician login failed: {clin_login.text}"
        clinician_headers = {"Authorization": f"Bearer {clin_login.json()['access_token']}"}

        print("--- STEP 1: Register Patient ---", flush=True)
        p_code = f"PAT-E2E-{os.urandom(3).hex().upper()}"
        p_res = await client.post("/api/v1/patients", json={
            "patient_code": p_code,
            "full_name": "Margaret Holloway",
            "date_of_birth": "1968-11-23",
            "sex": "F",
            "contact_email": "m.holloway@example.org"
        }, headers=admin_headers)
        assert p_res.status_code in [200, 201]
        pat_id = p_res.json()["id"]

        print("--- STEP 2: Create Screening Encounter ---", flush=True)
        enc_res = await client.post("/api/v1/screenings", json={
            "patient_id": pat_id,
            "notes": "Annual ophthalmology DR surveillance visit.",
            "age": 57,
            "diabetes_duration": 14.5,
            "hba1c": 8.6,
            "systolic_bp": 142.0,
            "diastolic_bp": 88.0,
            "bmi": 29.4
        }, headers=admin_headers)
        assert enc_res.status_code == 201
        enc_data = enc_res.json()
        assert enc_data["case_number"].startswith("SCR-")
        screening_id = enc_data["id"]

        print("--- STEP 3: Upload Fundus Image ---", flush=True)
        img = Image.new("RGB", (300, 300), color=(160, 50, 40))
        for x in range(80, 220, 10):
            for y in range(80, 220, 10):
                img.putpixel((x, y), (220, 180, 50))
                
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        buf.seek(0)

        upload_res = await client.post(
            f"/api/v1/screenings/{screening_id}/image",
            files={"file": ("fundus_eye.jpg", buf, "image/jpeg")},
            headers=admin_headers
        )
        assert upload_res.status_code == 200
        up_data = upload_res.json()
        assert up_data["quality_assessment"]["is_gradable"] is True
        assert "image_url" in up_data

        print("--- STEP 4: Predict & Grad-CAM ---", flush=True)
        pred_res = await client.post(f"/api/v1/screenings/{screening_id}/predict", headers=admin_headers)
        assert pred_res.status_code == 200
        pred_data = pred_res.json()
        assert pred_data["predicted_class"] in [0, 1, 2, 3, 4]
        assert 0.0 <= pred_data["confidence"] <= 1.0
        assert "uncertainty_level" in pred_data
        assert "gradcam_url" in pred_data
        assert "disclaimer" in pred_data
        original_ai_grade = pred_data["predicted_class"]

        print("--- STEP 5: Clinician Review ---", flush=True)
        review_res = await client.post(
            f"/api/v1/screenings/{screening_id}/review",
            json={
                "decision": "MODIFIED",
                "clinician_grade": (original_ai_grade + 1) % 5, # Deliberate override
                "clinician_notes": "Microaneurysms and multiple dot hemorrhages observed in macula. Grade modified by retina specialist.",
                "requires_followup": True,
                "followup_urgency": "Urgent"
            },
            headers=clinician_headers
        )
        assert review_res.status_code == 200
        rev_data = review_res.json()
        assert rev_data["decision"] == "MODIFIED"

        print("--- STEP 6: Verify AI Prediction Preservation ---", flush=True)
        detail_res = await client.get(f"/api/v1/screenings/{screening_id}", headers=clinician_headers)
        assert detail_res.status_code == 200
        detail = detail_res.json()
        assert detail["prediction"]["predicted_class"] == original_ai_grade, "AI prediction was erroneously overwritten!"
        assert detail["review"]["clinician_grade"] == (original_ai_grade + 1) % 5, "Clinician grade was not preserved!"

        print("--- STEP 7: Check Audit Trail ---", flush=True)
        audit_res = await client.get(f"/api/v1/screenings/{screening_id}/audit", headers=admin_headers)
        assert audit_res.status_code == 200
        audit_events = audit_res.json()
        actions = [a["action"] for a in audit_events]
        assert "ENCOUNTER_CREATED" in actions
        assert "IMAGE_UPLOADED" in actions
        assert "PREDICTION_GENERATED" in actions
        assert "CLINICIAN_REVIEW_SAVED" in actions

        print("--- STEP 8: Generate PDF Report ---", flush=True)
        pdf_res = await client.get(f"/api/v1/screenings/{screening_id}/report/pdf", headers=clinician_headers)
        assert pdf_res.status_code == 200
        assert pdf_res.headers["content-type"] == "application/pdf"
        assert len(pdf_res.content) > 1000 # Valid non-empty PDF bytes
        print("--- ALL WORKFLOW STEPS COMPLETE! ---", flush=True)
