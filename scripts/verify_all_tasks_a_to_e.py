import urllib.request
import urllib.error
import json
import io
import sys
from pathlib import Path

BASE_URL = "http://127.0.0.1:8000"

def run_checks():
    print("=" * 65)
    print("RUNNING COMPREHENSIVE VERIFICATION: TASKS A TO E")
    print("=" * 65)

    # -------------------------------------------------------------
    # TASK A: Authentication & Clinical Reports
    # -------------------------------------------------------------
    print("\n--- TASK A: Authentication & Clinical Reports ---")
    login_data = json.dumps({"email": "clinician@dr-screening.com", "password": "clinician123"}).encode()
    login_req = urllib.request.Request(
        f"{BASE_URL}/api/v1/auth/login",
        data=login_data,
        headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(login_req) as res:
        login_resp = json.loads(res.read().decode())
    
    token = login_resp.get("access_token")
    assert token, "Login failed: No access_token returned"
    print(f"[A.1] Login successful: Issued JWT token for {login_resp.get('user', {}).get('email')}")

    auth_headers = {"Authorization": f"Bearer {token}"}

    # Query Clinical Reports endpoint (GET /api/v1/screenings)
    rep_req = urllib.request.Request(f"{BASE_URL}/api/v1/screenings?limit=50", headers=auth_headers)
    with urllib.request.urlopen(rep_req) as res:
        screenings = json.loads(res.read().decode())
    
    assert isinstance(screenings, list), f"Expected list of screenings, got {type(screenings)}"
    assert len(screenings) > 0, "No screenings returned for Clinical Reports!"
    sample_sc = screenings[0]
    print(f"[A.2] Reports endpoint returned {len(screenings)} saved clinical screenings.")
    print(f"      Sample case: {sample_sc.get('case_number')} | Patient: {sample_sc.get('patient_code')} | Grade: {sample_sc.get('predicted_grade')} ({sample_sc.get('predicted_label')})")
    print("      TASK A STATUS: PASSED")

    # -------------------------------------------------------------
    # TASK B: Explainability (XAI) & Grad-CAM
    # -------------------------------------------------------------
    print("\n--- TASK B: Explainability (XAI) & Grad-CAM ---")
    # Fetch encounter details
    enc_id = sample_sc["id"]
    det_req = urllib.request.Request(f"{BASE_URL}/api/v1/screenings/{enc_id}", headers=auth_headers)
    with urllib.request.urlopen(det_req) as res:
        enc_detail = json.loads(res.read().decode())
    
    images = enc_detail.get("images", [])
    print(f"[B.1] Retrieved encounter {enc_detail['encounter']['case_number']} with {len(images)} retinal image(s).")
    assert len(images) > 0, f"Encounter {enc_id} has no retinal images!"
    print(f"      Image URL: {images[0]['image_url']}")

    # Request on-demand Grad-CAM generation
    g_req = urllib.request.Request(
        f"{BASE_URL}/api/v1/screenings/{enc_id}/gradcam?target_class=0",
        data=b"{}",
        headers={**auth_headers, "Content-Type": "application/json"}
    )
    with urllib.request.urlopen(g_req) as res:
        g_data = json.loads(res.read().decode())

    assert g_data.get("status") == "SUCCESS", f"Grad-CAM generation failed: {g_data}"
    assert g_data.get("gradcam_url"), "Grad-CAM URL missing in response"
    print(f"[B.2] Grad-CAM generated: {g_data['gradcam_url']} (Target Class: {g_data.get('class_name')})")
    print("      TASK B STATUS: PASSED")

    # -------------------------------------------------------------
    # TASK C: Research Metrics Audit
    # -------------------------------------------------------------
    print("\n--- TASK C: Research Metrics Audit ---")
    m_req = urllib.request.Request(f"{BASE_URL}/api/v1/analytics/research-metrics", headers=auth_headers)
    with urllib.request.urlopen(m_req) as res:
        metrics_resp = json.loads(res.read().decode())
    
    m = metrics_resp.get("metrics", {})
    print(f"[C.1] Metrics loaded:")
    print(f"      - Test Samples: {m.get('test_samples_count')} held-out fundus images")
    print(f"      - Test Accuracy: {m.get('accuracy') * 100:.1f}%")
    print(f"      - Quadratic Weighted Kappa: {m.get('qwk'):.3f}")
    print(f"      - Macro F1: {m.get('macro_f1'):.3f}")
    print(f"      - Class Distribution: {m.get('class_distribution')}")
    print(f"      - Validation Accuracy: {m.get('validation_metrics', {}).get('val_accuracy') * 100:.1f}%")
    print(f"      - Validation QWK: {m.get('validation_metrics', {}).get('val_qwk')}")
    assert m.get("test_samples_count") == 15, "Expected 15 test samples"
    print("      TASK C STATUS: PASSED")

    # -------------------------------------------------------------
    # TASK D: Federated Learning Simulation Status & Standby Nodes
    # -------------------------------------------------------------
    print("\n--- TASK D: Federated Learning Simulation & Standby Nodes ---")
    fl_req = urllib.request.Request(f"{BASE_URL}/api/v1/federated/status", headers=auth_headers)
    with urllib.request.urlopen(fl_req) as res:
        fl_status = json.loads(res.read().decode())

    nodes = fl_status.get("hospital_nodes", {})
    print(f"[D.1] FL Status: {fl_status.get('current_phase')} | Running: {fl_status.get('running')}")
    print(f"      Hospital Nodes ({len(nodes)}):")
    for k, v in sorted(nodes.items()):
        acc_str = f"{v.get('accuracy') * 100:.1f}%" if v.get('accuracy') is not None else "---"
        f1_str = f"{v.get('f1'):.3f}" if v.get('f1') is not None else "---"
        print(f"      - {k} ({v.get('name')}): Data Size = {v.get('data_size')} images | Accuracy = {acc_str} | F1 = {f1_str} | Status = {v.get('status')}")
    assert len(nodes) == 3, f"Expected 3 hospital nodes, got {len(nodes)}"
    for k, v in nodes.items():
        assert v.get("accuracy") is not None, f"Node {k} missing local accuracy!"
    print("      TASK D STATUS: PASSED")

    # -------------------------------------------------------------
    # TASK E: Administration & Model Registry
    # -------------------------------------------------------------
    print("\n--- TASK E: Administration & Model Registry ---")
    models_req = urllib.request.Request(f"{BASE_URL}/api/v1/models", headers=auth_headers)
    with urllib.request.urlopen(models_req) as res:
        model_list = json.loads(res.read().decode())
    
    print(f"[E.1] Registered Model Versions ({len(model_list)}):")
    assert len(model_list) >= 2, f"Expected at least 2 models, got {len(model_list)}"
    for mdl in model_list:
        print(f"      - [{mdl.get('id')[:8]}] {mdl.get('name')} ({mdl.get('architecture')})")
        print(f"        Active: {mdl.get('is_active')} | Status: {mdl.get('status')} | Checkpoint: {mdl.get('checkpoint_path')}")
        
        # Test physical verification endpoint
        v_req = urllib.request.Request(
            f"{BASE_URL}/api/v1/models/{mdl['id']}/verify",
            data=b"{}",
            headers={**auth_headers, "Content-Type": "application/json"}
        )
        with urllib.request.urlopen(v_req) as vres:
            vdata = json.loads(vres.read().decode())
        print(f"        Verification: Loadable = {vdata.get('loadable')} | Checksum = {vdata.get('checksum')} | Size = {vdata.get('file_size_bytes', 0) / (1024*1024):.2f} MB")
        assert vdata.get("valid") is True, f"Model verification failed for {mdl['name']}: {vdata}"
        assert vdata.get("loadable") is True, f"Model weights not loadable for {mdl['name']}: {vdata}"

    print("      TASK E STATUS: PASSED")

    print("\n" + "=" * 65)
    print("ALL 5 AUDIT TASKS (A, B, C, D, E) FULLY VERIFIED AND PASSING!")
    print("=" * 65)

if __name__ == "__main__":
    run_checks()
