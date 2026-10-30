import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fastapi.testclient import TestClient
from backend.main import app

def run_test():
    with TestClient(app) as client:
        print("1. Testing health...", flush=True)
        res = client.get("/health/live")
        print("Health live:", res.status_code, res.json(), flush=True)

        print("2. Testing login with JSON...", flush=True)
        res_login = client.post("/api/v1/auth/login", json={"email": "admin@dr-screening.com", "password": "admin123"})
        print("Login status:", res_login.status_code, flush=True)
        if res_login.status_code == 200:
            token = res_login.json()["access_token"]
            print("Token obtained successfully!", flush=True)
            
            print("3. Testing screenings list...", flush=True)
            headers = {"Authorization": f"Bearer {token}"}
            s_res = client.get("/api/v1/screenings", headers=headers)
            print("Screenings list status:", s_res.status_code, "Count:", len(s_res.json()), flush=True)

            print("4. Testing models active...", flush=True)
            m_res = client.get("/api/v1/models/active", headers=headers)
            print("Active model:", m_res.status_code, m_res.json().get("name"), flush=True)
        else:
            print("Login failed:", res_login.text, flush=True)

if __name__ == "__main__":
    run_test()
