import os
import json
from pathlib import Path
from typing import Dict, Any, List
from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, desc

from ..database import get_db
from ..models.patient import Patient
from ..models.prediction import Prediction
from ..models.review import ClinicalReview
from ..schemas.prediction import CLASS_NAMES

router = APIRouter(prefix="/analytics", tags=["analytics"])
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


@router.get("/dashboard-stats")
@router.get("/dashboard")
async def get_dashboard_stats(db: AsyncSession = Depends(get_db)) -> Dict[str, Any]:
    """Calculates live database statistics for the main dashboard."""
    # 1. Total Patients
    res_pat = await db.execute(select(func.count(Patient.id)))
    total_patients = res_pat.scalar() or 0

    # 2. Total Predictions
    res_pred = await db.execute(select(func.count(Prediction.id)))
    total_predictions = res_pred.scalar() or 0

    # 3. Pending Reviews
    res_pend = await db.execute(
        select(func.count(Prediction.id)).where(Prediction.review_status == "PENDING")
    )
    pending_reviews = res_pend.scalar() or 0

    # 4. Completed Reviews & Agreement Rate
    res_rev = await db.execute(select(func.count(ClinicalReview.id)))
    completed_reviews = res_rev.scalar() or 0

    res_agreed = await db.execute(
        select(func.count(ClinicalReview.id)).where(ClinicalReview.agreed_with_ai == True)
    )
    agreed_reviews = res_agreed.scalar() or 0
    agreement_rate = (agreed_reviews / completed_reviews * 100) if completed_reviews > 0 else 0.0

    # 5. Class distribution from database
    class_counts = {c: 0 for c in range(5)}
    res_dist = await db.execute(
        select(Prediction.predicted_class, func.count(Prediction.id)).group_by(Prediction.predicted_class)
    )
    for c_id, count in res_dist.all():
        if c_id in class_counts:
            class_counts[c_id] = count

    distribution_data = [
        {"name": CLASS_NAMES[i], "class_id": i, "value": class_counts[i]}
        for i in range(5)
    ]

    return {
        "total_patients": total_patients,
        "total_predictions": total_predictions,
        "pending_reviews": pending_reviews,
        "completed_reviews": completed_reviews,
        "agreement_rate": round(agreement_rate, 1),
        "class_distribution": distribution_data
    }


@router.get("/research-metrics")
@router.get("/research")
async def get_research_metrics() -> Dict[str, Any]:
    """Loads actual test set evaluation results from saved experiment logs."""
    eval_file = PROJECT_ROOT / "logs" / "test_evaluation_results.json"
    if eval_file.exists():
        try:
            with open(eval_file, "r") as f:
                data = json.load(f)
            return {
                "available": True,
                "metrics": data
            }
        except Exception:
            pass

    # Check federated results
    fed_file = PROJECT_ROOT / "outputs" / "federated_results.json"
    if fed_file.exists():
        try:
            with open(fed_file, "r") as f:
                data = json.load(f)
            fed_metrics = data.get("federated", {})
            return {
                "available": True,
                "metrics": fed_metrics,
                "source": "federated_experiment"
            }
        except Exception:
            pass

    return {
        "available": False,
        "message": "No evaluation results available. Run model evaluation or federated simulation first."
    }


@router.post("/evaluate")
async def trigger_model_evaluation():
    """
    Runs live evaluation of the active PyTorch CNN checkpoint against the held-out test split.
    Calculates test accuracy, per-class precision, recall, F1-score, specificity,
    quadratic weighted kappa (QWK), AUC-ROC, and confusion matrix.
    Contrasts test metrics with validation metrics from the checkpoint.
    """
    try:
        from scripts.evaluate_model import run_test_evaluation
        metrics = run_test_evaluation(
            checkpoint_path=str(PROJECT_ROOT / "checkpoints" / "best_cnn_model.pth"),
            data_dir=str(PROJECT_ROOT / "data" / "retinal"),
            output_dir=str(PROJECT_ROOT / "logs")
        )
        return {
            "available": True,
            "status": "SUCCESS",
            "message": "Held-out test split evaluation computed successfully.",
            "metrics": metrics
        }
    except Exception as e:
        return {
            "available": False,
            "status": "ERROR",
            "message": f"Model evaluation failed: {str(e)}"
        }
