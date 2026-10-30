from typing import List, Dict, Any, Optional
from datetime import datetime, timezone
from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, desc

from ..database import get_db
from ..models.user import User
from ..models.patient import Patient
from ..models.prediction import Prediction
from ..models.review import ClinicalReview
from ..models.screening import ScreeningEncounter, RetinalImage
from ..schemas.review import ReviewCreate, ReviewOut, ReviewUpdate
from ..schemas.prediction import PredictionResponse, CLASS_NAMES
from ..auth import get_current_active_user, log_audit_event

router = APIRouter(prefix="/reviews", tags=["reviews"])


@router.get("/pending", response_model=List[PredictionResponse])
async def list_pending_reviews(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=100),
    db: AsyncSession = Depends(get_db)
):
    """
    Returns full screening predictions that have NOT been reviewed yet
    (review_status is PENDING or no ClinicalReview attached).
    """
    stmt = (
        select(Prediction, Patient.patient_code, Patient.full_name)
        .outerjoin(Patient, Prediction.patient_id == Patient.id)
        .outerjoin(ClinicalReview, Prediction.id == ClinicalReview.prediction_id)
        .where(ClinicalReview.id == None)
        .order_by(desc(Prediction.created_at))
        .offset(skip)
        .limit(limit)
    )
    result = await db.execute(stmt)
    rows = result.all()

    pending_list = []
    for pred, p_code, p_name in rows:
        po = PredictionResponse.model_validate(pred)
        po.prediction = pred.predicted_label
        po.patient_code = p_code
        po.patient_name = p_name
        if pred.class_probabilities and len(pred.class_probabilities) == 5:
            po.probabilities = {
                CLASS_NAMES[i]: float(pred.class_probabilities[i]) for i in range(5)
            }
        if pred.image_path:
            import os
            po.image_url = f"/api/v1/uploads/{os.path.basename(pred.image_path)}"
        if pred.gradcam_path:
            import os
            po.gradcam_url = f"/api/v1/uploads/{os.path.basename(pred.gradcam_path)}"
            po.gradcam_available = True
        pending_list.append(po)
    return pending_list


@router.post("", response_model=ReviewOut)
@router.post("/", response_model=ReviewOut)
async def create_review(
    review_in: ReviewCreate, 
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """
    Submits clinician review (Approval, Modification, or Rejection).
    Updates review status without altering original AI prediction values.
    """
    # 1. Fetch prediction
    stmt = select(Prediction).where(Prediction.id == review_in.prediction_id)
    result = await db.execute(stmt)
    pred = result.scalar_one_or_none()
    if not pred:
        raise HTTPException(status_code=404, detail="Prediction not found")

    # 2. Check if already reviewed
    stmt = select(ClinicalReview).where(ClinicalReview.prediction_id == review_in.prediction_id)
    result = await db.execute(stmt)
    if result.scalar_one_or_none():
        raise HTTPException(status_code=400, detail="Review already exists for this screening")

    # 3. Create review record
    db_review = ClinicalReview(
        prediction_id=review_in.prediction_id,
        reviewer_id=current_user.id,
        agreed_with_ai=review_in.agreed_with_ai,
        decision=review_in.decision,
        clinician_grade=review_in.clinician_grade if review_in.clinician_grade is not None else pred.predicted_class,
        clinician_notes=review_in.clinician_notes,
        final_diagnosis=review_in.final_diagnosis or (CLASS_NAMES[review_in.clinician_grade] if review_in.clinician_grade is not None else pred.predicted_label),
        requires_followup=review_in.requires_followup,
        followup_urgency=review_in.followup_urgency or "routine",
        review_timestamp=datetime.now(timezone.utc)
    )

    # 4. Update prediction status
    pred.review_status = review_in.decision

    # 5. Synchronize associated ScreeningEncounter status if linked
    if getattr(pred, "encounter_id", None):
        e_stmt = select(ScreeningEncounter).where(ScreeningEncounter.id == pred.encounter_id)
        e_res = await db.execute(e_stmt)
        enc = e_res.scalar_one_or_none()
        if enc:
            enc.status = review_in.decision
    elif pred.image_path:
        ri_stmt = select(RetinalImage).where(RetinalImage.file_path == pred.image_path).limit(1)
        ri_res = await db.execute(ri_stmt)
        ri = ri_res.scalar_one_or_none()
        if ri:
            e_stmt = select(ScreeningEncounter).where(ScreeningEncounter.id == ri.encounter_id)
            e_res = await db.execute(e_stmt)
            enc = e_res.scalar_one_or_none()
            if enc:
                enc.status = review_in.decision
                pred.encounter_id = enc.id

    db.add(db_review)
    await db.commit()
    await db.refresh(db_review)

    # 6. Log audit event
    await log_audit_event(
        db=db,
        action="CLINICIAN_REVIEW_SAVED",
        resource_type="screening",
        resource_id=pred.id,
        user=current_user,
        details={
            "prediction_id": pred.id,
            "decision": db_review.decision,
            "clinician_grade": db_review.clinician_grade,
            "final_diagnosis": db_review.final_diagnosis,
            "agreed_with_ai": db_review.agreed_with_ai,
            "reviewer": current_user.full_name
        }
    )

    return db_review


@router.get("", response_model=List[ReviewOut])
@router.get("/", response_model=List[ReviewOut])
async def list_reviews(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=100),
    db: AsyncSession = Depends(get_db)
):
    stmt = select(ClinicalReview).order_by(desc(ClinicalReview.review_timestamp)).offset(skip).limit(limit)
    result = await db.execute(stmt)
    return result.scalars().all()


@router.get("/stats")
async def review_stats(db: AsyncSession = Depends(get_db)) -> Dict[str, Any]:
    stmt = select(func.count(ClinicalReview.id))
    result = await db.execute(stmt)
    total = result.scalar() or 0

    if total == 0:
        return {"total": 0, "agreed": 0, "disagreed": 0, "agreement_rate": 0.0}

    stmt = select(func.count(ClinicalReview.id)).where(ClinicalReview.agreed_with_ai == True)
    result = await db.execute(stmt)
    agreed = result.scalar() or 0

    return {
        "total": total,
        "agreed": agreed,
        "disagreed": total - agreed,
        "agreement_rate": round(agreed / total, 3)
    }


@router.get("/{id}", response_model=ReviewOut)
async def get_review(id: str, db: AsyncSession = Depends(get_db)):
    stmt = select(ClinicalReview).where(ClinicalReview.id == id)
    result = await db.execute(stmt)
    review = result.scalar_one_or_none()
    if not review:
        raise HTTPException(status_code=404, detail="Review not found")
    return review
