from typing import Dict, Any
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from ..auth import get_current_active_user

router = APIRouter(prefix="/federated", tags=["federated"])

class FLStartRequest(BaseModel):
    rounds: int = Field(default=3, ge=1, le=20, description="Number of federated communication rounds")
    clients: int = Field(default=3, ge=1, le=10, description="Number of simulated hospital client nodes")
    local_epochs: int = Field(default=1, ge=1, le=10, description="Local training epochs per round per node")
    batch_size: int = Field(default=16, ge=1, le=128, description="Batch size for training")
    lr: float = Field(default=0.0001, ge=1e-6, le=1.0, description="Learning rate for local node optimizer")

@router.get("/status")
async def get_fl_status() -> Dict[str, Any]:
    from ..services.fl_service import get_fl_service
    fl_service = get_fl_service()
    return fl_service.get_status()

@router.post("/start")
@router.post("/simulate")
async def start_fl(req: FLStartRequest, current_user = Depends(get_current_active_user)):
    from ..services.fl_service import get_fl_service
    fl_service = get_fl_service()
    started, msg = fl_service.start_simulation(req.model_dump())
    if not started:
        raise HTTPException(status_code=400, detail=msg)
        
    return {
        "message": "Federated Learning simulation started successfully",
        "run_id": msg,
        "config": req.model_dump()
    }

@router.get("/results")
async def get_fl_results() -> Dict[str, Any]:
    from ..services.fl_service import get_fl_service
    fl_service = get_fl_service()
    return fl_service.get_results()

@router.get("/history")
async def get_fl_history() -> Any:
    fl_service = get_fl_service()
    status = fl_service.get_status()
    return status.get("metrics_history", [])

@router.get("/comparison")
async def get_fl_comparison() -> Dict[str, Any]:
    fl_service = get_fl_service()
    results = fl_service.get_results()
    return {
        "centralized": results.get("centralized", {}),
        "federated": results.get("federated", results.get("federated_final", {}))
    }

