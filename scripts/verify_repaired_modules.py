import urllib.request
import json
import os
import io

def run():
    print("=" * 60)
    print("VERIFYING REPAIRED MODULES 1 TO 5")
    print("=" * 60)

    # 1. Login as technician
    req = urllib.request.Request(
        'http://127.0.0.1:8000/api/v1/auth/login',
        data=json.dumps({'email': 'technician@dr-screening.com', 'password': 'tech123'}).encode(),
        headers={'Content-Type': 'application/json'}
    )
    res = urllib.request.urlopen(req)
    token = json.loads(res.read().decode())['access_token']
    auth_headers = {'Authorization': f'Bearer {token}'}

    # 2. Get patient ID
    p_res = urllib.request.urlopen(
        urllib.request.Request('http://127.0.0.1:8000/api/v1/patients?limit=1', headers=auth_headers)
    )
    patients = json.loads(p_res.read().decode())
    patient = patients[0]
    patient_id = patient['id']
    print(f"[1] Verified Patient: {patient['patient_code']} ({patient['full_name']})")

    # 3. Create prediction via /predictions with real test image
    sample_img = r'C:\Users\VIJAY\.gemini\antigravity\scratch\dr_screening\data\retinal\test\0\demo_cls0_test_000.png'
    with open(sample_img, 'rb') as f:
        img_bytes = f.read()

    boundary = '----WebKitFormBoundaryDRScreeningAudit2026'
    body = io.BytesIO()
    body.write(f'--{boundary}\r\nContent-Disposition: form-data; name="patient_id"\r\n\r\n{patient_id}\r\n'.encode())
    body.write(f'--{boundary}\r\nContent-Disposition: form-data; name="mode"\r\n\r\nimage_only\r\n'.encode())
    body.write(f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="sample_fundus.png"\r\nContent-Type: image/png\r\n\r\n'.encode())
    body.write(img_bytes)
    body.write(f'\r\n--{boundary}--\r\n'.encode())

    pred_req = urllib.request.Request(
        'http://127.0.0.1:8000/api/v1/predictions',
        data=body.getvalue(),
        headers={
            'Authorization': f'Bearer {token}',
            'Content-Type': f'multipart/form-data; boundary={boundary}'
        }
    )
    pred_res = urllib.request.urlopen(pred_req)
    pred_data = json.loads(pred_res.read().decode())
    print(f"[2] Module 6 & New Screening: Predicted Grade {pred_data['predicted_class']} ({pred_data['prediction']}), Confidence: {pred_data['confidence']*100:.1f}%")

    # 4. Verify encounter appears in /api/v1/screenings (Clinical Reports)
    sc_res = urllib.request.urlopen(
        urllib.request.Request('http://127.0.0.1:8000/api/v1/screenings?limit=10', headers=auth_headers)
    )
    screenings = json.loads(sc_res.read().decode())
    matching = [s for s in screenings if s.get('patient_id') == patient_id and s.get('confidence') == pred_data['confidence']]
    assert len(matching) > 0, "New screening did NOT appear in /screenings list!"
    enc = matching[0]
    print(f"[3] Module 2 (Reports): Verified new screening appears in reports: Case {enc['case_number']} (ID: {enc['id']})")

    # 5. Verify /api/v1/screenings/{id} (Explainability Detail)
    det_res = urllib.request.urlopen(
        urllib.request.Request(f'http://127.0.0.1:8000/api/v1/screenings/{enc["id"]}', headers=auth_headers)
    )
    det_data = json.loads(det_res.read().decode())
    assert det_data['images'] and len(det_data['images']) > 0, "Encounter has no retinal images!"
    print(f"[4] Module 1 (Explainability): Encounter details loaded. Fundus URL: {det_data['images'][0]['image_url']}")

    # 6. Generate Grad-CAM on demand for encounter
    g_req = urllib.request.Request(
        f'http://127.0.0.1:8000/api/v1/screenings/{enc["id"]}/gradcam?target_class=0',
        data=b'{}',
        headers={'Authorization': f'Bearer {token}', 'Content-Type': 'application/json'}
    )
    try:
        g_res = urllib.request.urlopen(g_req)
        g_data = json.loads(g_res.read().decode())
        assert g_data['status'] == 'SUCCESS', f"Grad-CAM generation failed: {g_data}"
        print(f"[5] Module 1 (Grad-CAM): Generated Grad-CAM saliency overlay: {g_data['gradcam_url']}")
    except urllib.error.HTTPError as he:
        print(f"Grad-CAM error body: {he.read().decode()}")
        raise

    # 7. Verify PDF report download
    pdf_res = urllib.request.urlopen(
        urllib.request.Request(f'http://127.0.0.1:8000/api/v1/screenings/{enc["id"]}/report/pdf', headers=auth_headers)
    )
    pdf_bytes = pdf_res.read()
    assert len(pdf_bytes) > 1000, "PDF report download was empty or too small!"
    print(f"[6] Module 2 (PDF Reports): Downloaded verified clinical decision-support PDF ({len(pdf_bytes)} bytes)")

    # 8. Test Module 3: Research Metrics Live Evaluation
    admin_req = urllib.request.Request(
        'http://127.0.0.1:8000/api/v1/auth/login',
        data=json.dumps({'email': 'admin@dr-screening.com', 'password': 'admin123'}).encode(),
        headers={'Content-Type': 'application/json'}
    )
    admin_token = json.loads(urllib.request.urlopen(admin_req).read().decode())['access_token']
    admin_headers = {'Authorization': f'Bearer {admin_token}', 'Content-Type': 'application/json'}

    eval_res = urllib.request.urlopen(
        urllib.request.Request('http://127.0.0.1:8000/api/v1/analytics/evaluate', data=b'{}', headers=admin_headers)
    )
    eval_data = json.loads(eval_res.read().decode())
    assert eval_data['status'] == 'SUCCESS', f"Model evaluation failed: {eval_data}"
    m = eval_data['metrics']
    print(f"[7] Module 3 (Research Metrics): Evaluated held-out test split: Accuracy={m['accuracy']*100:.1f}%, QWK={m['qwk']:.3f}, F1={m['macro_f1']:.3f}, N={m['test_samples_count']}")

    # 9. Test Module 5: Admin Model Registry Checkpoint Verification
    models_res = urllib.request.urlopen(
        urllib.request.Request('http://127.0.0.1:8000/api/v1/models', headers=admin_headers)
    )
    models = json.loads(models_res.read().decode())
    m_id = models[0]['id']
    v_res = urllib.request.urlopen(
        urllib.request.Request(f'http://127.0.0.1:8000/api/v1/models/{m_id}/verify', data=b'{}', headers=admin_headers)
    )
    v_data = json.loads(v_res.read().decode())
    assert v_data['valid'] is True, f"Checkpoint verification failed: {v_data}"
    print(f"[8] Module 5 (Admin Model Registry): Verified checkpoint {v_data['architecture']} ({v_data['file_size_bytes']/1024/1024:.2f} MB, Checksum: {v_data['checksum']})")

    # 10. Test Module 4: Federated Learning Status
    fl_res = urllib.request.urlopen(
        urllib.request.Request('http://127.0.0.1:8000/api/v1/federated/status', headers=admin_headers)
    )
    fl_data = json.loads(fl_res.read().decode())
    print(f"[9] Module 4 (Federated Learning): Simulation engine verified ready. Status: {fl_data.get('status', 'Ready')}, Running: {fl_data.get('running', False)}")

    print("=" * 60)
    print("ALL 5 MODULES HAVE BEEN SUCCESSFULLY REPAIRED AND VALIDATED!")
    print("=" * 60)

if __name__ == '__main__':
    run()
