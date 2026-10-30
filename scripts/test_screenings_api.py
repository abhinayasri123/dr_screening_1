import sys
import os
from pathlib import Path

BASE_DIR = Path("C:/Users/VIJAY/.gemini/antigravity/scratch/dr_screening")
sys.path.insert(0, str(BASE_DIR))
os.chdir(str(BASE_DIR))

import asyncio
from httpx import AsyncClient, ASGITransport
from backend.main import app

async def test_api():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        # 1. Login to get token
        login_res = await ac.post("/api/v1/auth/login", data={"username": "dr.smith@hospital.org", "password": "Password123!"})
        print("Login status:", login_res.status_code)
        if login_res.status_code != 200:
            print("Login response:", login_res.text)
            return
        
        token = login_res.json()["access_token"]
        headers = {"Authorization": f"Bearer {token}"}
        
        # 2. Call /api/v1/screenings
        sc_res = await ac.get("/api/v1/screenings?limit=20", headers=headers)
        print("Screenings status:", sc_res.status_code)
        print("Screenings type:", type(sc_res.json()))
        data = sc_res.json()
        print("Total items returned:", len(data))
        if data:
            print("First item keys:", list(data[0].keys()))
            print("First item:", data[0])
            
            # 3. Call /api/v1/screenings/{id}
            first_id = data[0]["id"]
            detail_res = await ac.get(f"/api/v1/screenings/{first_id}", headers=headers)
            print("\nDetail status for", first_id, ":", detail_res.status_code)
            detail = detail_res.json()
            print("Detail keys:", list(detail.keys()))
            print("Detail images:", len(detail.get("images", [])))
            print("Detail prediction exists:", detail.get("prediction") is not None)
            if detail.get("images"):
                print("Image 0:", detail["images"][0])
            if detail.get("prediction"):
                print("Prediction:", detail["prediction"]["predicted_class"], detail["prediction"]["predicted_label"], detail["prediction"]["gradcam_url"])

if __name__ == "__main__":
    asyncio.run(test_api())
