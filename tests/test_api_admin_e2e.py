import sys
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from fastapi.testclient import TestClient
from backend.main import app

def test_admin_api():
    client = TestClient(app)

    # 1. Login as clinician
    resp = client.post("/api/v1/auth/login", json={"email": "clinician@dr-screening.com", "password": "clinician123"})
    assert resp.status_code == 200, f"Login failed: {resp.text}"
    token = resp.json().get("access_token")
    assert token, "No access token in response"
    headers = {"Authorization": f"Bearer {token}"}
    print("[PASS] Login as clinician succeeded.")

    # 2. Test /admin/users
    r_users = client.get("/api/v1/admin/users", headers=headers)
    assert r_users.status_code == 200, f"Failed /admin/users: {r_users.text}"
    users = r_users.json()
    assert len(users) >= 4, f"Expected at least 4 users, got {len(users)}"
    print(f"[PASS] /admin/users returned {len(users)} users.")

    # 3. Test /admin/audit-logs/stats
    r_stats = client.get("/api/v1/admin/audit-logs/stats", headers=headers)
    assert r_stats.status_code == 200, f"Failed /admin/audit-logs/stats: {r_stats.text}"
    stats = r_stats.json()
    assert "total_events" in stats, "Missing total_events in stats"
    print(f"[PASS] /admin/audit-logs/stats returned total_events={stats['total_events']}.")

    # 4. Test /admin/audit-logs
    r_logs = client.get("/api/v1/admin/audit-logs?limit=10", headers=headers)
    assert r_logs.status_code == 200, f"Failed /admin/audit-logs: {r_logs.text}"
    logs = r_logs.json()
    assert "items" in logs and "total" in logs, "Invalid audit-logs response format"
    assert len(logs["items"]) > 0, "Expected at least 1 audit event in items"
    print(f"[PASS] /admin/audit-logs returned {len(logs['items'])} items (total={logs['total']}).")

    # 5. Test /admin/audit-logs/verify
    r_verify = client.post("/api/v1/admin/audit-logs/verify", json={}, headers=headers)
    assert r_verify.status_code == 200, f"Failed /admin/audit-logs/verify: {r_verify.text}"
    verify_res = r_verify.json()
    assert verify_res.get("status") == "VERIFIED", f"Verification failed: {verify_res}"
    assert verify_res.get("chain_valid") is True, f"Chain invalid: {verify_res}"
    print(f"[PASS] /admin/audit-logs/verify: status={verify_res.get('status')}, chain_valid={verify_res.get('chain_valid')}.")

    # 6. Test Model Registry
    r_models = client.get("/api/v1/models", headers=headers)
    assert r_models.status_code == 200, f"Failed /models: {r_models.text}"
    models = r_models.json()
    assert len(models) > 0, "No models returned"
    print(f"[PASS] /models returned {len(models)} model versions.")

    print("\nALL ADMIN API END-TO-END TESTS PASSED!")

if __name__ == "__main__":
    test_admin_api()
