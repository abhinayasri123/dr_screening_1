import os
import sys



# Prevent network timeouts and matplotlib font scanning hangs
os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"
_cache_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".cache", "matplotlib"))
os.makedirs(_cache_dir, exist_ok=True)
os.environ["MPLCONFIGDIR"] = _cache_dir

from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from .database import create_tables, get_db, async_sessionmaker_factory
from .auth import get_password_hash
from . import models
from .models.user import User
from .models.model_registry import ModelVersion
from .models.patient import Patient
from .models.screening import ScreeningEncounter, RetinalImage
from .services.inference_service import get_inference_service
from .schemas.prediction import MEDICAL_DISCLAIMER

from .routers import auth, patients, predictions, reviews, federated, metrics, screenings, models_admin

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Ensure uploads and reports directories exist
    upload_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "uploads"))
    reports_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "reports"))
    os.makedirs(upload_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)

    # Initialize schema if not disabled by test runners
    if os.getenv("SKIP_CREATE_TABLES") != "1":
        try:
            await create_tables()
        except Exception:
            pass

    yield

app = FastAPI(
    title="Privacy-Preserving Diabetic Retinopathy Screening Platform",
    description=f"Clinical decision-support API combining CNN-based Retinal Classification, Federated Learning, Explainable AI, and Clinician-in-the-Loop Review.\\n\\n{MEDICAL_DISCLAIMER}",
    version="2.0.0",
    lifespan=lifespan
)

# CORS configuration supporting local dev and deployment domains
allowed_origins_env = os.getenv("ALLOWED_ORIGINS", "")
allowed_origins = [o.strip() for o in allowed_origins_env.split(",") if o.strip()]
if not allowed_origins:
    allowed_origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Core Routers
app.include_router(auth.router, prefix="/api/v1")
app.include_router(patients.router, prefix="/api/v1")
app.include_router(screenings.router, prefix="/api/v1")
app.include_router(predictions.router, prefix="/api/v1")
app.include_router(reviews.router, prefix="/api/v1")
app.include_router(federated.router, prefix="/api/v1")
app.include_router(metrics.router, prefix="/api/v1")
app.include_router(metrics.router, prefix="/api/v1/metrics")
app.include_router(models_admin.router, prefix="/api/v1")

# Mount static uploads
upload_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "uploads"))
os.makedirs(upload_dir, exist_ok=True)
app.mount("/api/v1/uploads", StaticFiles(directory=upload_dir), name="uploads")

# Health endpoints compliant with Section 14
@app.get("/health/live")
@app.get("/api/v1/health/live")
async def health_liveness():
    return {"status": "alive", "service": "healthcare-dr-screening-api"}

@app.get("/health/ready")
@app.get("/api/v1/health/ready")
async def health_readiness():
    inf_service = get_inference_service()
    is_ready = bool(inf_service.cnn_model is not None or getattr(inf_service, "model", None) is not None)
    return {
        "status": "ready" if is_ready else "initializing",
        "model_loaded": is_ready,
        "database": "connected"
    }

@app.get("/api/v1/health")
async def health_check():
    inf_service = get_inference_service()
    is_loaded = bool(inf_service.cnn_model is not None or getattr(inf_service, "model", None) is not None)
    return {
        "status": "healthy",
        "model_loaded": is_loaded,
        "demo_mode": inf_service.is_demo_mode,
        "active_architecture": "EfficientNet-B3",
        "federated_learning": "ready"
    }

# ── Serve Built Frontend SPA (Single URL Integration) ──
frontend_dist = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "frontend", "dist"))
if os.path.exists(frontend_dist):
    assets_dir = os.path.join(frontend_dist, "assets")
    if os.path.exists(assets_dir):
        app.mount("/assets", StaticFiles(directory=assets_dir), name="frontend-assets")
    
    videos_dist = os.path.join(frontend_dist, "videos")
    if not os.path.exists(videos_dist):
        videos_dist = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "frontend", "public", "videos"))
    if os.path.exists(videos_dist):
        app.mount("/videos", StaticFiles(directory=videos_dist), name="frontend-videos")
    
    @app.get("/{full_path:path}")
    async def serve_spa(full_path: str):
        # Don't intercept API routes or uploads
        if full_path.startswith("api/") or full_path.startswith("docs") or full_path.startswith("openapi.json") or full_path.startswith("health"):
            raise HTTPException(status_code=404, detail="Not found")
        file_path = os.path.join(frontend_dist, full_path)
        if os.path.isfile(file_path):
            return FileResponse(file_path)
        return FileResponse(os.path.join(frontend_dist, "index.html"))
