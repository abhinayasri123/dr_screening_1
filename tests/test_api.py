import pytest
import asyncio
from functools import wraps
import httpx
from backend.main import app

def async_test(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        return asyncio.run(f(*args, **kwargs))
    return wrapper

@async_test
async def test_health_endpoints():
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        res_live = await client.get("/health/live")
        assert res_live.status_code == 200
        assert res_live.json()["status"] == "alive"

        res_ready = await client.get("/health/ready")
        assert res_ready.status_code == 200
        assert "status" in res_ready.json()

        res_health = await client.get("/api/v1/health")
        assert res_health.status_code == 200
        data = res_health.json()
        assert data["status"] == "healthy"
        assert "model_loaded" in data

@async_test
async def test_login_flow_roles():
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        roles = [
            ("admin@dr-screening.com", "admin123", "admin"),
            ("clinician@dr-screening.com", "clinician123", "clinician"),
            ("technician@dr-screening.com", "tech123", "technician"),
            ("researcher@dr-screening.com", "research123", "researcher"),
        ]
        for email, pwd, expected_role in roles:
            response = await client.post("/api/v1/auth/login", json={
                "email": email,
                "password": pwd
            })
            assert response.status_code == 200, f"Login failed for {email}: {response.text}"
            data = response.json()
            assert "access_token" in data
            assert data["token_type"] == "bearer"
            assert data["user"]["role"] == expected_role

@async_test
async def test_login_failure():
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        response = await client.post("/api/v1/auth/login", json={
            "email": "wrong@example.com",
            "password": "wrongpassword"
        })
        assert response.status_code == 401

@async_test
async def test_model_registry_endpoints():
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        res = await client.post("/api/v1/auth/login", json={"email": "admin@dr-screening.com", "password": "admin123"})
        token = res.json()["access_token"]
        headers = {"Authorization": f"Bearer {token}"}

        res_models = await client.get("/api/v1/models", headers=headers)
        assert res_models.status_code == 200
        models = res_models.json()
        assert isinstance(models, list)
        assert len(models) >= 1

        res_active = await client.get("/api/v1/models/active", headers=headers)
        assert res_active.status_code == 200
        assert res_active.json()["is_active"] is True
