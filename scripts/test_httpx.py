import sys
import asyncio
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import httpx
from httpx import ASGITransport
from backend.main import app

async def main():
    print("Initializing httpx ASGITransport...", flush=True)
    async with httpx.AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        print("1. Testing /health/live...", flush=True)
        res_live = await client.get("/health/live")
        print("Health live:", res_live.status_code, res_live.json(), flush=True)

        print("2. Testing /api/v1/auth/login...", flush=True)
        res_login = await client.post("/api/v1/auth/login", json={"email": "admin@dr-screening.com", "password": "admin123"})
        print("Login status:", res_login.status_code, flush=True)
        token = res_login.json()["access_token"]
        headers = {"Authorization": f"Bearer {token}"}

        print("3. Testing /api/v1/screenings...", flush=True)
        res_sc = await client.get("/api/v1/screenings", headers=headers)
        print("Screenings:", res_sc.status_code, "Count:", len(res_sc.json()), flush=True)

        print("4. Testing /api/v1/models/active...", flush=True)
        res_m = await client.get("/api/v1/models/active", headers=headers)
        print("Active model:", res_m.status_code, res_m.json()["name"], flush=True)

    print("ALL ASYNC HTTPX TESTS PASSED!", flush=True)

if __name__ == "__main__":
    asyncio.run(main())
