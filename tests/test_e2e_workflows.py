import io
import os
import pytest
from PIL import Image
from fastapi.testclient import TestClient
from backend.main import app

@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c

@pytest.fixture(scope="module")
def auth_headers(client):
    res = client.post("/api/v1/auth/login", data={
        "username": "admin@dr-screening.com",
        "password": "admin123"
    })
    assert res.status_code == 200, "Admin login failed in test fixture"
    token = res.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}

def test_system_health(client):
    res = client.get("/api/v1/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "healthy"
    assert "model_loaded" in data

def test_patient_crud_and_persistence(client, auth_headers):
    # Create patient
    import uuid
    code = f"TEST-PAT-{uuid.uuid4().hex[:6].upper()}"
    patient_payload = {
        "patient_code": code,
        "full_name": "Eleanor Vance (Test)",
        "date_of_birth": "1972-04-18",
        "sex": "F",
        "contact_email": "eleanor.test@ophthalmology.org"
    }
    create_res = client.post("/api/v1/patients", json=patient_payload, headers=auth_headers)
    assert create_res.status_code in [200, 201]
    created_pat = create_res.json()
    assert created_pat["patient_code"] == code
    pat_id = created_pat["id"]

    # Retrieve patient
    get_res = client.get(f"/api/v1/patients/{pat_id}", headers=auth_headers)
    assert get_res.status_code == 200
    assert get_res.json()["patient_code"] == code

    # List patients
    list_res = client.get("/api/v1/patients", headers=auth_headers)
    assert list_res.status_code == 200
    patients = list_res.json()
    found = any(p["id"] == pat_id for p in patients)
    assert found, "Created patient was not found in patient list"

def test_retinal_image_upload_and_real_inference(client, auth_headers):
    # Create a synthetic retinal fundus test image in-memory
    img = Image.new("RGB", (224, 224), color=(180, 50, 40))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    buf.seek(0)

    # First get or create a test patient
    p_res = client.get("/api/v1/patients", headers=auth_headers)
    patients = p_res.json()
    if patients:
        pat_id = patients[0]["id"]
    else:
        create_res = client.post("/api/v1/patients", json={
            "patient_code": "P-AUTO-01",
            "full_name": "Automated Test Subject",
            "date_of_birth": "1980-01-01",
            "sex": "M"
        }, headers=auth_headers)
        pat_id = create_res.json()["id"]

    # Post multipart image to /api/v1/predictions
    files = {"file": ("test_retina.jpg", buf, "image/jpeg")}
    data = {"patient_id": pat_id, "mode": "image_only"}

    pred_res = client.post("/api/v1/predictions", files=files, data=data, headers=auth_headers)
    assert pred_res.status_code == 200, f"Prediction failed: {pred_res.text}"
    p_data = pred_res.json()

    # Validate 5-class distribution
    assert "predicted_class" in p_data
    assert p_data["predicted_class"] in [0, 1, 2, 3, 4]
    assert "confidence" in p_data
    assert 0.0 <= p_data["confidence"] <= 1.0
    assert "probabilities" in p_data
    probs = p_data["probabilities"]
    assert len(probs) == 5
    assert abs(sum(probs.values()) - 1.0) < 0.05
    assert p_data["review_status"] == "PENDING"
    assert "gradcam_url" in p_data

def test_clinician_review_workflow(client, auth_headers):
    # List pending reviews
    pend_res = client.get("/api/v1/reviews/pending", headers=auth_headers)
    assert pend_res.status_code == 200
    pending_items = pend_res.json()
    assert len(pending_items) > 0, "Expected at least one pending review from previous test"

    target_pred = pending_items[0]
    pred_id = target_pred["id"]

    # Submit clinician review decision
    review_payload = {
        "prediction_id": pred_id,
        "agreed_with_ai": True,
        "decision": "APPROVED",
        "clinician_grade": target_pred["predicted_class"],
        "clinician_notes": "Lesion patterns confirmed on optic disc and macula. Validated by ophthalmology reviewer.",
        "final_diagnosis": target_pred.get("predicted_label", "Moderate DR"),
        "requires_followup": False,
        "followup_urgency": "routine"
    }

    sub_res = client.post("/api/v1/reviews", json=review_payload, headers=auth_headers)
    assert sub_res.status_code in [200, 201], f"Review submission failed: {sub_res.text}"

    # Confirm it is no longer in pending queue
    pend_after = client.get("/api/v1/reviews/pending", headers=auth_headers).json()
    assert not any(p["id"] == pred_id for p in pend_after), "Reviewed item still appeared in pending queue"

    # Confirm completed reviews contains it
    comp_after = client.get("/api/v1/reviews", headers=auth_headers).json()
    assert any(c["prediction_id"] == pred_id for c in comp_after), "Review record not found in completed reviews"

def test_dashboard_statistics(client, auth_headers):
    res = client.get("/api/v1/analytics/dashboard-stats", headers=auth_headers)
    assert res.status_code == 200
    data = res.json()
    assert "total_patients" in data
    assert "total_predictions" in data
    assert "pending_reviews" in data
    assert "completed_reviews" in data
    assert data["total_patients"] >= 1
    assert data["total_predictions"] >= 1

def test_research_metrics_endpoint(client, auth_headers):
    res = client.get("/api/v1/analytics/research-metrics")
    assert res.status_code == 200
    data = res.json()
    assert "available" in data
