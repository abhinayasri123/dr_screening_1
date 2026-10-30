"""
Screening Encounters & Clinician Workflow Router
================================================
Implements the core clinician-in-the-loop screening workflow:
- Registering screening encounters with non-identifying case references (SCR-YYYYMMDD-XXXX)
- Image upload and real-time automated quality screening
- CNN model inference and Grad-CAM generation
- Clinician review queue, decision recording, override logging
- PDF clinical report generation
- Audit trail per case
"""

import os
import uuid
import csv
import io
from datetime import datetime, timezone
from typing import List, Optional, Dict, Any

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form, Query, Response
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, desc, func

from ..database import get_db
from ..models.user import User
from ..models.patient import Patient
from ..models.screening import ScreeningEncounter, RetinalImage, ClinicalMetadata
from ..models.prediction import Prediction
from ..models.review import ClinicalReview
from ..models.audit import AuditEvent
from ..auth import (
    get_current_active_user,
    require_clinician,
    require_technician,
    require_admin,
    log_audit_event
)
from ..services.inference_service import get_inference_service, CLASS_NAMES
from ..utils.image_utils import save_upload_file, validate_image_file
from ml.datasets.quality_checker import assess_image_quality

router = APIRouter(prefix="/screenings", tags=["screenings"])

from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
REPORTS_DIR = os.path.abspath(PROJECT_ROOT / "reports")
UPLOAD_DIR = os.path.abspath(PROJECT_ROOT / "uploads")
os.makedirs(REPORTS_DIR, exist_ok=True)
os.makedirs(UPLOAD_DIR, exist_ok=True)

# ----------------- Pydantic Schemas -----------------
class ScreeningCreate(BaseModel):
    patient_id: str
    notes: Optional[str] = None
    age: Optional[int] = None
    diabetes_duration: Optional[float] = None
    hba1c: Optional[float] = None
    systolic_bp: Optional[float] = None
    diastolic_bp: Optional[float] = None
    bmi: Optional[float] = None
    creatinine: Optional[float] = None
    medical_history: Optional[str] = None

class ClinicianReviewRequest(BaseModel):
    decision: str = Field(..., description="APPROVED, MODIFIED, REJECTED, UNGRADABLE, REPEAT_REQUESTED, ESCALATED")
    clinician_grade: Optional[int] = Field(None, ge=0, le=4)
    clinician_notes: Optional[str] = None
    requires_followup: bool = False
    followup_urgency: str = Field("Routine", description="Routine, Urgent, Emergency")

# ----------------- Helper Functions -----------------
def generate_case_reference() -> str:
    now_str = datetime.now(timezone.utc).strftime("%Y%m%d")
    rand_suffix = uuid.uuid4().hex[:4].upper()
    return f"SCR-{now_str}-{rand_suffix}"

# ----------------- Endpoints -----------------

@router.post("", status_code=201)
@router.post("/", status_code=201)
async def create_screening(
    data: ScreeningCreate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """Step 1 & 2: Create a screening encounter with a unique non-identifying case reference."""
    stmt = select(Patient).where(Patient.id == data.patient_id)
    res = await db.execute(stmt)
    patient = res.scalar_one_or_none()
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")

    case_ref = generate_case_reference()
    encounter = ScreeningEncounter(
        case_number=case_ref,
        patient_id=data.patient_id,
        status="AWAITING_IMAGE",
        notes=data.notes,
        created_by=current_user.id
    )
    db.add(encounter)
    await db.flush()

    # Save optional clinical metadata if provided
    has_metadata = any(v is not None for v in [
        data.age, data.diabetes_duration, data.hba1c,
        data.systolic_bp, data.diastolic_bp, data.bmi, data.creatinine
    ])
    if has_metadata:
        c_meta = ClinicalMetadata(
            encounter_id=encounter.id,
            patient_id=data.patient_id,
            age=data.age,
            diabetes_duration=data.diabetes_duration,
            hba1c=data.hba1c,
            systolic_bp=data.systolic_bp,
            diastolic_bp=data.diastolic_bp,
            bmi=data.bmi,
            creatinine=data.creatinine,
            medical_history=data.medical_history
        )
        db.add(c_meta)

    await db.commit()
    await db.refresh(encounter)

    await log_audit_event(
        db=db,
        action="ENCOUNTER_CREATED",
        resource_type="screening",
        resource_id=encounter.id,
        user=current_user,
        details={"case_number": case_ref, "patient_id": data.patient_id}
    )

    return {
        "id": encounter.id,
        "case_number": encounter.case_number,
        "patient_id": encounter.patient_id,
        "status": encounter.status,
        "encounter_date": encounter.encounter_date.isoformat(),
        "message": "Screening encounter created successfully. Ready for retinal fundus image upload."
    }

@router.get("")
@router.get("/")
async def list_screenings(
    status: Optional[str] = Query(None),
    search: Optional[str] = Query(None),
    has_image: Optional[bool] = Query(None),
    limit: int = Query(50, le=100),
    offset: int = Query(0),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """List screening encounters with status filtering, search, and image availability verification."""
    stmt = select(ScreeningEncounter).order_by(desc(ScreeningEncounter.encounter_date))
    if status and status.upper() != "ALL":
        stmt = stmt.where(ScreeningEncounter.status == status.upper())
    if search:
        stmt = stmt.where(ScreeningEncounter.case_number.ilike(f"%{search}%"))

    stmt = stmt.offset(offset).limit(limit)
    res = await db.execute(stmt)
    encounters = res.scalars().all()

    results = []
    for enc in encounters:
        p_stmt = select(Patient).where(Patient.id == enc.patient_id)
        p_res = await db.execute(p_stmt)
        pat = p_res.scalar_one_or_none()

        img_stmt = select(RetinalImage).where(RetinalImage.encounter_id == enc.id).order_by(desc(RetinalImage.uploaded_at)).limit(1)
        img_res = await db.execute(img_stmt)
        retinal_img = img_res.scalar_one_or_none()

        pred = None
        # Match prediction strictly by encounter_id first, then by retinal image file path
        pred_stmt = select(Prediction).where(Prediction.encounter_id == enc.id).order_by(desc(Prediction.created_at)).limit(1)
        pred_res = await db.execute(pred_stmt)
        pred = pred_res.scalar_one_or_none()
        if not pred and retinal_img:
            pred_stmt = select(Prediction).where(Prediction.image_path == retinal_img.file_path).order_by(desc(Prediction.created_at)).limit(1)
            pred_res = await db.execute(pred_stmt)
            pred = pred_res.scalar_one_or_none()

        rev = None
        if pred:
            rev_stmt = select(ClinicalReview).where(ClinicalReview.prediction_id == pred.id)
            rev_res = await db.execute(rev_stmt)
            rev = rev_res.scalar_one_or_none()

        # Image existence validation
        img_url = None
        has_verified_image = False
        effective_img_p = retinal_img.file_path if retinal_img else (pred.image_path if pred else None)
        if effective_img_p:
            if os.path.exists(effective_img_p):
                img_url = f"/api/v1/uploads/{os.path.basename(effective_img_p)}"
                has_verified_image = True
            else:
                candidate = os.path.join(UPLOAD_DIR, os.path.basename(effective_img_p))
                if os.path.exists(candidate):
                    img_url = f"/api/v1/uploads/{os.path.basename(effective_img_p)}"
                    has_verified_image = True

        gc_url = None
        has_gc = False
        if pred and pred.gradcam_path:
            if os.path.exists(pred.gradcam_path):
                gc_url = f"/api/v1/uploads/{os.path.basename(pred.gradcam_path)}"
                has_gc = True
            else:
                candidate_gc = os.path.join(UPLOAD_DIR, os.path.basename(pred.gradcam_path))
                if os.path.exists(candidate_gc):
                    gc_url = f"/api/v1/uploads/{os.path.basename(pred.gradcam_path)}"
                    has_gc = True

        if has_image is True and not has_verified_image:
            continue

        results.append({
            "id": enc.id,
            "case_number": enc.case_number,
            "patient_id": enc.patient_id,
            "patient_code": pat.patient_code if pat else "N/A",
            "patient_name": pat.full_name if pat else "Unknown",
            "status": enc.status,
            "encounter_date": enc.encounter_date.isoformat(),
            "has_image": has_verified_image,
            "image_url": img_url,
            "has_gradcam": has_gc,
            "gradcam_url": gc_url,
            "predicted_grade": pred.predicted_class if pred else None,
            "predicted_label": pred.predicted_label if pred else None,
            "confidence": pred.confidence if pred else None,
            "uncertainty_level": pred.uncertainty_level if pred else None,
            "class_probabilities": pred.class_probabilities if pred else None,
            "review_decision": rev.decision if rev else None,
            "review_grade": rev.clinician_grade if rev else None,
            "notes": enc.notes
        })

    return results

@router.get("/xai/cases")
async def list_xai_eligible_cases(
    limit: int = Query(50, le=100),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """
    Specifically returns screening cases that have verified fundus images on disk,
    ready for Explainability / Grad-CAM analysis.
    Prioritizes cases with predictions, reviews, or active fundus photographs.
    """
    stmt = (
        select(ScreeningEncounter)
        .order_by(desc(ScreeningEncounter.encounter_date))
        .limit(limit * 2)
    )
    res = await db.execute(stmt)
    encounters = res.scalars().all()

    eligible = []
    for enc in encounters:
        p_stmt = select(Patient).where(Patient.id == enc.patient_id)
        p_res = await db.execute(p_stmt)
        pat = p_res.scalar_one_or_none()

        img_stmt = select(RetinalImage).where(RetinalImage.encounter_id == enc.id).order_by(desc(RetinalImage.uploaded_at)).limit(1)
        img_res = await db.execute(img_stmt)
        retinal_img = img_res.scalar_one_or_none()

        pred = None
        pred_stmt = select(Prediction).where(Prediction.encounter_id == enc.id).order_by(desc(Prediction.created_at)).limit(1)
        pred_res = await db.execute(pred_stmt)
        pred = pred_res.scalar_one_or_none()
        if not pred and retinal_img:
            pred_stmt = select(Prediction).where(Prediction.image_path == retinal_img.file_path).order_by(desc(Prediction.created_at)).limit(1)
            pred_res = await db.execute(pred_stmt)
            pred = pred_res.scalar_one_or_none()

        effective_img_p = retinal_img.file_path if retinal_img else (pred.image_path if pred else None)
        if not effective_img_p:
            continue

        resolved_img = None
        if os.path.exists(effective_img_p):
            resolved_img = effective_img_p
        else:
            candidate = os.path.join(UPLOAD_DIR, os.path.basename(effective_img_p))
            if os.path.exists(candidate):
                resolved_img = candidate

        if not resolved_img:
            continue  # Must have physical image on disk

        gc_url = None
        if pred and pred.gradcam_path:
            if os.path.exists(pred.gradcam_path) or os.path.exists(os.path.join(UPLOAD_DIR, os.path.basename(pred.gradcam_path))):
                gc_url = f"/api/v1/uploads/{os.path.basename(pred.gradcam_path)}"

        eligible.append({
            "id": enc.id,
            "case_number": enc.case_number,
            "patient_id": enc.patient_id,
            "patient_code": pat.patient_code if pat else "N/A",
            "patient_name": pat.full_name if pat else "Unknown",
            "status": enc.status,
            "encounter_date": enc.encounter_date.isoformat(),
            "has_image": True,
            "image_url": f"/api/v1/uploads/{os.path.basename(resolved_img)}",
            "has_gradcam": bool(gc_url),
            "gradcam_url": gc_url,
            "predicted_grade": pred.predicted_class if pred else None,
            "predicted_label": pred.predicted_label if pred else "Pending AI Inference",
            "confidence": pred.confidence if pred else None,
            "uncertainty_level": pred.uncertainty_level if pred else "Normal",
            "class_probabilities": pred.class_probabilities if pred else None,
        })
        if len(eligible) >= limit:
            break

    return eligible

@router.get("/{screening_id}")
async def get_screening_detail(
    screening_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """Retrieve full detail for a screening encounter."""
    stmt = select(ScreeningEncounter).where(ScreeningEncounter.id == screening_id)
    res = await db.execute(stmt)
    enc = res.scalar_one_or_none()
    if not enc:
        # Fallback: check if screening_id is a prediction ID
        p_stmt = select(Prediction).where(Prediction.id == screening_id)
        p_res = await db.execute(p_stmt)
        matched_p = p_res.scalar_one_or_none()
        if matched_p:
            if matched_p.encounter_id:
                e_stmt = select(ScreeningEncounter).where(ScreeningEncounter.id == matched_p.encounter_id)
                e_res = await db.execute(e_stmt)
                enc = e_res.scalar_one_or_none()
            if not enc:
                e_stmt = select(ScreeningEncounter).where(ScreeningEncounter.patient_id == matched_p.patient_id).order_by(desc(ScreeningEncounter.encounter_date)).limit(1)
                e_res = await db.execute(e_stmt)
                enc = e_res.scalar_one_or_none()

    if not enc:
        raise HTTPException(status_code=404, detail="Screening encounter not found")

    p_stmt = select(Patient).where(Patient.id == enc.patient_id)
    p_res = await db.execute(p_stmt)
    pat = p_res.scalar_one_or_none()

    # Images
    img_stmt = select(RetinalImage).where(RetinalImage.encounter_id == enc.id).order_by(desc(RetinalImage.uploaded_at))
    img_res = await db.execute(img_stmt)
    images = img_res.scalars().all()

    # Prediction: match strictly by encounter_id first, then by image_path
    pred = None
    pred_stmt = select(Prediction).where(Prediction.encounter_id == enc.id).order_by(desc(Prediction.created_at)).limit(1)
    pred_res = await db.execute(pred_stmt)
    pred = pred_res.scalar_one_or_none()
    if not pred and images:
        pred_stmt = select(Prediction).where(Prediction.image_path == images[0].file_path).order_by(desc(Prediction.created_at)).limit(1)
        pred_res = await db.execute(pred_stmt)
        pred = pred_res.scalar_one_or_none()

    # Review
    rev = None
    if pred:
        rev_stmt = select(ClinicalReview).where(ClinicalReview.prediction_id == pred.id)
        rev_res = await db.execute(rev_stmt)
        rev = rev_res.scalar_one_or_none()

    # Clinical Metadata
    meta_stmt = select(ClinicalMetadata).where(ClinicalMetadata.encounter_id == enc.id)
    meta_res = await db.execute(meta_stmt)
    meta = meta_res.scalar_one_or_none()

    images_payload = [
        {
            "id": im.id,
            "filename": im.filename,
            "file_path": im.file_path,
            "image_url": f"/api/v1/uploads/{os.path.basename(im.file_path)}",
            "blur_score": im.blur_score,
            "mean_luminance": im.mean_luminance,
            "is_gradable": im.is_gradable,
            "quality_status": im.quality_status
        } for im in images
    ]
    if not images_payload and pred and pred.image_path and os.path.exists(pred.image_path):
        q = assess_image_quality(pred.image_path)
        images_payload = [
            {
                "id": f"pred-{pred.id}",
                "filename": pred.image_filename or os.path.basename(pred.image_path),
                "file_path": pred.image_path,
                "image_url": f"/api/v1/uploads/{os.path.basename(pred.image_path)}",
                "blur_score": q.get("blur_score", 0.0),
                "mean_luminance": q.get("mean_luminance", 0.0),
                "is_gradable": q.get("is_gradable", False),
                "quality_status": q.get("quality_status", "UNKNOWN")
            }
        ]

    return {
        "encounter": {
            "id": enc.id,
            "case_number": enc.case_number,
            "patient_id": enc.patient_id,
            "status": enc.status,
            "encounter_date": enc.encounter_date.isoformat(),
            "notes": enc.notes
        },
        "patient": {
            "id": pat.id if pat else None,
            "patient_code": pat.patient_code if pat else "N/A",
            "full_name": pat.full_name if pat else "Unknown",
            "sex": pat.sex if pat else "N/A",
            "date_of_birth": pat.date_of_birth.isoformat() if pat and pat.date_of_birth else None
        },
        "images": images_payload,
        "prediction": {
            "id": pred.id,
            "predicted_class": pred.predicted_class,
            "predicted_label": pred.predicted_label,
            "class_probabilities": pred.class_probabilities,
            "confidence": pred.confidence,
            "uncertainty_std": pred.uncertainty_std,
            "uncertainty_level": pred.uncertainty_level,
            "gradcam_url": f"/api/v1/uploads/{os.path.basename(pred.gradcam_path)}" if pred.gradcam_path else None,
            "model_version": pred.model_version,
            "created_at": pred.created_at.isoformat()
        } if pred else None,
        "review": {
            "id": rev.id,
            "decision": rev.decision,
            "clinician_grade": rev.clinician_grade,
            "clinician_notes": rev.clinician_notes,
            "final_diagnosis": rev.final_diagnosis,
            "requires_followup": rev.requires_followup,
            "followup_urgency": rev.followup_urgency,
            "review_timestamp": rev.review_timestamp.isoformat() if rev.review_timestamp else None
        } if rev else None,
        "clinical_metadata": {
            "age": meta.age,
            "diabetes_duration": meta.diabetes_duration,
            "hba1c": meta.hba1c,
            "systolic_bp": meta.systolic_bp,
            "diastolic_bp": meta.diastolic_bp,
            "bmi": meta.bmi,
            "creatinine": meta.creatinine,
            "medical_history": meta.medical_history,
            "status_notice": "Clinical risk prediction model not configured (research prototype)"
        } if meta else None
    }

@router.post("/{screening_id}/image")
async def upload_screening_image(
    screening_id: str,
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """Step 3 & 4: Upload retinal fundus photograph and evaluate image quality."""
    stmt = select(ScreeningEncounter).where(ScreeningEncounter.id == screening_id)
    res = await db.execute(stmt)
    enc = res.scalar_one_or_none()
    if not enc:
        raise HTTPException(status_code=404, detail="Screening encounter not found")

    if not validate_image_file(file):
        raise HTTPException(status_code=400, detail="Invalid file format. Please upload PNG, JPG, or JPEG.")

    # Save to uploads directory
    saved_path = save_upload_file(file)

    # Perform automated image quality assessment & retinal authenticity verification (Stage A)
    from ml.datasets.quality_checker import assess_image_quality, EXACT_INVALID_IMAGE_MESSAGE
    quality = assess_image_quality(saved_path)

    if not quality.get("is_retinal", True):
        # Clean up rejected non-retinal file
        if os.path.exists(saved_path):
            try:
                os.remove(saved_path)
            except Exception:
                pass
        raise HTTPException(
            status_code=422,
            detail=EXACT_INVALID_IMAGE_MESSAGE
        )

    # Valid retinal fundus image verified. Record retinal image.
    retinal_img = RetinalImage(
        encounter_id=enc.id,
        file_path=saved_path,
        filename=file.filename or "fundus.jpg",
        width=quality["resolution"][0],
        height=quality["resolution"][1],
        blur_score=quality["blur_score"],
        mean_luminance=quality["mean_luminance"],
        is_gradable=quality["is_gradable"],
        quality_status=quality["quality_status"]
    )
    db.add(retinal_img)

    # Encounter is ready for prediction regardless of borderline optical quality
    enc.status = "AWAITING_PREDICTION"

    await db.commit()
    await db.refresh(retinal_img)

    await log_audit_event(
        db=db,
        action="IMAGE_UPLOADED",
        resource_type="screening",
        resource_id=enc.id,
        user=current_user,
        details={
            "filename": file.filename,
            "quality_status": quality["quality_status"],
            "blur_score": quality["blur_score"],
            "is_gradable": quality["is_gradable"]
        }
    )

    return {
        "image_id": retinal_img.id,
        "image_url": f"/api/v1/uploads/{os.path.basename(saved_path)}",
        "quality_assessment": quality,
        "encounter_status": enc.status,
        "message": "Image uploaded and quality evaluated."
    }

@router.post("/{screening_id}/predict")
async def run_screening_prediction(
    screening_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """Step 5 & 6: Run CNN model inference and Grad-CAM for the uploaded fundus image."""
    stmt = select(ScreeningEncounter).where(ScreeningEncounter.id == screening_id)
    res = await db.execute(stmt)
    enc = res.scalar_one_or_none()
    if not enc:
        raise HTTPException(status_code=404, detail="Screening encounter not found")

    img_stmt = select(RetinalImage).where(RetinalImage.encounter_id == enc.id).order_by(desc(RetinalImage.uploaded_at)).limit(1)
    img_res = await db.execute(img_stmt)
    retinal_img = img_res.scalar_one_or_none()
    if not retinal_img:
        raise HTTPException(status_code=400, detail="No fundus image uploaded for this screening encounter.")

    # Run actual CNN inference (if image has borderline optical quality, it will be marked INCONCLUSIVE)
    inf_service = get_inference_service()
    try:
        pred_result = inf_service.predict(image_path=retinal_img.file_path, mode="image_only")
    except ValueError as ve:
        raise HTTPException(status_code=422, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference pipeline execution error: {str(e)}")

    # Generate Grad-CAM saliency overlay
    gradcam_path = None
    if inf_service.gradcam_explainer:
        try:
            gradcam_path = inf_service.generate_gradcam(
                image_path=retinal_img.file_path,
                target_class=pred_result.predicted_class
            )
        except Exception as e:
            print(f"[Screening] Grad-CAM generation warning: {e}")

    prediction = Prediction(
        encounter_id=enc.id,
        patient_id=enc.patient_id,
        image_path=retinal_img.file_path,
        image_filename=retinal_img.filename,
        prediction_mode="image_only",
        predicted_class=pred_result.predicted_class,
        predicted_label=pred_result.predicted_label,
        class_probabilities=pred_result.class_probabilities,
        confidence=pred_result.confidence,
        uncertainty_std=pred_result.uncertainty_std,
        uncertainty_level=pred_result.uncertainty_level,
        gradcam_path=gradcam_path,
        model_version=inf_service._model_version,
        is_demo_mode=inf_service.is_demo_mode,
        review_status="PENDING",
        created_by=current_user.id
    )
    db.add(prediction)

    # Update encounter status to AWAITING_REVIEW
    enc.status = "AWAITING_REVIEW"
    await db.commit()
    await db.refresh(prediction)

    await log_audit_event(
        db=db,
        action="PREDICTION_GENERATED",
        resource_type="screening",
        resource_id=enc.id,
        user=current_user,
        details={
            "predicted_class": pred_result.predicted_class,
            "confidence": pred_result.confidence,
            "uncertainty_level": pred_result.uncertainty_level,
            "gradcam_generated": bool(gradcam_path)
        }
    )

    return {
        "prediction_id": prediction.id,
        "predicted_class": prediction.predicted_class,
        "predicted_label": prediction.predicted_label,
        "confidence": prediction.confidence,
        "uncertainty_std": prediction.uncertainty_std,
        "uncertainty_level": prediction.uncertainty_level,
        "summary_status": pred_result.summary_status,
        "is_gradable": pred_result.is_gradable,
        "quality_status": pred_result.quality_status,
        "quality_warnings": pred_result.quality_warnings,
        "clinical_recommendation": pred_result.clinical_recommendation,
        "probabilities": {CLASS_NAMES[i]: prediction.class_probabilities[i] for i in range(5)},
        "gradcam_url": f"/api/v1/uploads/{os.path.basename(gradcam_path)}" if gradcam_path else None,
        "encounter_status": enc.status,
        "disclaimer": pred_result.disclaimer
    }

@router.post("/{screening_id}/gradcam")
async def generate_screening_gradcam(
    screening_id: str,
    target_class: Optional[int] = Query(None, ge=0, le=4),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """Generate or regenerate Grad-CAM saliency overlay for an encounter's retinal image."""
    stmt = select(ScreeningEncounter).where(ScreeningEncounter.id == screening_id)
    res = await db.execute(stmt)
    enc = res.scalar_one_or_none()
    
    # Fallback: check if screening_id is a prediction ID
    pred = None
    if not enc:
        p_stmt = select(Prediction).where(Prediction.id == screening_id)
        p_res = await db.execute(p_stmt)
        pred = p_res.scalar_one_or_none()
        if pred:
            e_stmt = select(ScreeningEncounter).where(ScreeningEncounter.patient_id == pred.patient_id).order_by(desc(ScreeningEncounter.encounter_date)).limit(1)
            e_res = await db.execute(e_stmt)
            enc = e_res.scalar_one_or_none()

    if not enc and not pred:
        raise HTTPException(status_code=404, detail="Screening encounter or prediction not found")

    retinal_img = None
    if enc:
        img_stmt = select(RetinalImage).where(RetinalImage.encounter_id == enc.id).order_by(desc(RetinalImage.uploaded_at)).limit(1)
        img_res = await db.execute(img_stmt)
        retinal_img = img_res.scalar_one_or_none()

    # Match prediction if not already found
    if not pred and enc:
        pred_stmt = select(Prediction).where(Prediction.encounter_id == enc.id).order_by(desc(Prediction.created_at)).limit(1)
        pred_res = await db.execute(pred_stmt)
        pred = pred_res.scalar_one_or_none()
    if not pred and retinal_img:
        pred_stmt = select(Prediction).where(Prediction.image_path == retinal_img.file_path).order_by(desc(Prediction.created_at)).limit(1)
        pred_res = await db.execute(pred_stmt)
        pred = pred_res.scalar_one_or_none()

    effective_image_path = retinal_img.file_path if retinal_img else (pred.image_path if pred else None)
    if not effective_image_path or not os.path.exists(effective_image_path):
        raise HTTPException(status_code=400, detail="No fundus image file found on server for this screening.")

    inf_service = get_inference_service()
    cls_to_explain = target_class if target_class is not None else (pred.predicted_class if pred else 2)

    gradcam_filename = f"gradcam_{os.path.basename(effective_image_path)}"
    gradcam_path = os.path.join(UPLOAD_DIR, gradcam_filename)

    try:
        inf_service.generate_gradcam(
            image_path=effective_image_path,
            target_class=cls_to_explain,
            save_path=gradcam_path
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Grad-CAM generation failed: {str(e)}")

    if pred:
        pred.gradcam_path = gradcam_path
        await db.commit()

    return {
        "status": "SUCCESS",
        "gradcam_url": f"/api/v1/uploads/{os.path.basename(gradcam_path)}",
        "target_class": cls_to_explain,
        "class_name": CLASS_NAMES[cls_to_explain]
    }

@router.post("/{screening_id}/review")
async def review_screening(
    screening_id: str,
    data: ClinicianReviewRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(require_clinician)
):
    """Step 8 & 9: Authorized clinician accepts, adjusts, or rejects AI suggestion with audit logging."""
    stmt = select(ScreeningEncounter).where(ScreeningEncounter.id == screening_id)
    res = await db.execute(stmt)
    enc = res.scalar_one_or_none()
    if not enc:
        raise HTTPException(status_code=404, detail="Screening encounter not found")

    # Match prediction strictly by encounter_id first, then by retinal image file path
    pred_stmt = select(Prediction).where(Prediction.encounter_id == enc.id).order_by(desc(Prediction.created_at)).limit(1)
    pred_res = await db.execute(pred_stmt)
    pred = pred_res.scalar_one_or_none()
    if not pred:
        img_stmt = select(RetinalImage).where(RetinalImage.encounter_id == enc.id).order_by(desc(RetinalImage.uploaded_at)).limit(1)
        img_res = await db.execute(img_stmt)
        r_img = img_res.scalar_one_or_none()
        if r_img:
            pred_stmt = select(Prediction).where(Prediction.image_path == r_img.file_path).order_by(desc(Prediction.created_at)).limit(1)
            pred_res = await db.execute(pred_stmt)
            pred = pred_res.scalar_one_or_none()
    if not pred:
        raise HTTPException(status_code=400, detail="Cannot review a screening encounter with no AI prediction.")

    agreed = (data.decision == "APPROVED")
    final_diag = CLASS_NAMES[data.clinician_grade] if data.clinician_grade is not None else (
        pred.predicted_label if agreed else "Clinician Overridden Diagnosis"
    )

    # Check if review already exists
    rev_stmt = select(ClinicalReview).where(ClinicalReview.prediction_id == pred.id)
    rev_res = await db.execute(rev_stmt)
    review = rev_res.scalar_one_or_none()

    if not review:
        review = ClinicalReview(
            prediction_id=pred.id,
            reviewer_id=current_user.id,
            agreed_with_ai=agreed,
            decision=data.decision,
            clinician_grade=data.clinician_grade if data.clinician_grade is not None else pred.predicted_class,
            clinician_notes=data.clinician_notes,
            final_diagnosis=final_diag,
            requires_followup=data.requires_followup,
            followup_urgency=data.followup_urgency
        )
        db.add(review)
    else:
        review.reviewer_id = current_user.id
        review.agreed_with_ai = agreed
        review.decision = data.decision
        review.clinician_grade = data.clinician_grade if data.clinician_grade is not None else pred.predicted_class
        review.clinician_notes = data.clinician_notes
        review.final_diagnosis = final_diag
        review.requires_followup = data.requires_followup
        review.followup_urgency = data.followup_urgency
        review.review_timestamp = datetime.now(timezone.utc)

    # Status transition
    pred.review_status = data.decision
    enc.status = data.decision

    await db.commit()
    await db.refresh(review)

    await log_audit_event(
        db=db,
        action="CLINICIAN_REVIEW_SAVED",
        resource_type="screening",
        resource_id=enc.id,
        user=current_user,
        details={
            "decision": data.decision,
            "clinician_grade": review.clinician_grade,
            "agreed_with_ai": agreed,
            "reviewer": current_user.full_name
        }
    )

    return {
        "status": "SUCCESS",
        "review_id": review.id,
        "encounter_status": enc.status,
        "decision": review.decision,
        "confirmed_grade": review.clinician_grade,
        "final_diagnosis": review.final_diagnosis,
        "message": "Clinician decision recorded separately from original AI prediction."
    }

@router.get("/{screening_id}/audit")
async def get_screening_audit_trail(
    screening_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """Retrieve immutable audit trail for the screening encounter."""
    stmt = select(AuditEvent).where(AuditEvent.resource_id == screening_id).order_by(AuditEvent.created_at)
    res = await db.execute(stmt)
    events = res.scalars().all()
    return [
        {
            "id": ev.id,
            "action": ev.action,
            "user_email": ev.user_email,
            "details": ev.details,
            "timestamp": ev.created_at.isoformat()
        } for ev in events
    ]

@router.get("/{screening_id}/report")
@router.get("/{screening_id}/report/pdf")
async def download_screening_pdf(
    screening_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """Generate and stream publication-quality PDF clinical decision-support report."""
    stmt = select(ScreeningEncounter).where(ScreeningEncounter.id == screening_id)
    res = await db.execute(stmt)
    enc = res.scalar_one_or_none()
    if not enc:
        raise HTTPException(status_code=404, detail="Screening encounter not found")

    p_stmt = select(Patient).where(Patient.id == enc.patient_id)
    p_res = await db.execute(p_stmt)
    pat = p_res.scalar_one_or_none()

    # Image
    img_stmt = select(RetinalImage).where(RetinalImage.encounter_id == enc.id).order_by(desc(RetinalImage.uploaded_at)).limit(1)
    img_res = await db.execute(img_stmt)
    retinal_img = img_res.scalar_one_or_none()

    # Prediction: match strictly by encounter_id first, then by image_path
    pred_stmt = select(Prediction).where(Prediction.encounter_id == enc.id).order_by(desc(Prediction.created_at)).limit(1)
    pred_res = await db.execute(pred_stmt)
    pred = pred_res.scalar_one_or_none()
    if not pred and retinal_img:
        pred_stmt = select(Prediction).where(Prediction.image_path == retinal_img.file_path).order_by(desc(Prediction.created_at)).limit(1)
        pred_res = await db.execute(pred_stmt)
        pred = pred_res.scalar_one_or_none()

    if not pred:
        raise HTTPException(status_code=400, detail="Cannot generate report before AI prediction is executed.")

    # Review
    rev_stmt = select(ClinicalReview).where(ClinicalReview.prediction_id == pred.id)
    rev_res = await db.execute(rev_stmt)
    rev = rev_res.scalar_one_or_none()

    # User reviewer
    rev_name = "Pending"
    if rev:
        u_stmt = select(User).where(User.id == rev.reviewer_id)
        u_res = await db.execute(u_stmt)
        u = u_res.scalar_one_or_none()
        if u:
            rev_name = u.full_name

    pdf_filename = f"Report_{enc.case_number}.pdf"
    pdf_path = os.path.join(REPORTS_DIR, pdf_filename)

    from ..services.pdf_report_service import generate_screening_pdf
    generate_screening_pdf(
        report_output_path=pdf_path,
        case_number=enc.case_number,
        patient_code=pat.patient_code if pat else "ANON-001",
        encounter_date=enc.encounter_date.strftime("%Y-%m-%d %H:%M UTC"),
        prediction_data={
            "predicted_class": pred.predicted_class,
            "predicted_label": pred.predicted_label,
            "confidence": pred.confidence,
            "uncertainty_std": pred.uncertainty_std,
            "uncertainty_level": pred.uncertainty_level,
            "class_probabilities": pred.class_probabilities
        },
        quality_data={
            "quality_status": retinal_img.quality_status if retinal_img else "GRADABLE",
            "blur_score": retinal_img.blur_score if retinal_img else 120.0,
            "mean_luminance": retinal_img.mean_luminance if retinal_img else 85.0
        },
        review_data={
            "status": enc.status,
            "decision": rev.decision if rev else "PENDING_VERIFICATION",
            "clinician_grade": rev.clinician_grade if rev else "Pending",
            "clinician_notes": rev.clinician_notes if rev else "Case awaiting ophthalmologist review.",
            "reviewer_name": rev_name,
            "followup_urgency": rev.followup_urgency if rev else "Routine",
            "review_timestamp": rev.review_timestamp.strftime("%Y-%m-%d %H:%M UTC") if rev and rev.review_timestamp else "Pending"
        },
        original_image_path=retinal_img.file_path if retinal_img else None,
        gradcam_image_path=pred.gradcam_path,
        model_version=pred.model_version
    )

    await log_audit_event(
        db=db,
        action="REPORT_DOWNLOADED",
        resource_type="screening",
        resource_id=enc.id,
        user=current_user,
        details={"case_number": enc.case_number, "format": "PDF"}
    )

    return FileResponse(
        path=pdf_path,
        filename=pdf_filename,
        media_type="application/pdf"
    )

@router.get("/export/csv")
async def export_screenings_csv(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """Export authorized aggregate screening records as CSV."""
    stmt = select(ScreeningEncounter).order_by(desc(ScreeningEncounter.encounter_date))
    res = await db.execute(stmt)
    encounters = res.scalars().all()

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow([
        "case_number", "encounter_date", "status",
        "predicted_grade", "predicted_label", "confidence", "uncertainty_level",
        "clinician_decision", "clinician_grade"
    ])

    for enc in encounters:
        pred_stmt = select(Prediction).where(Prediction.patient_id == enc.patient_id).order_by(desc(Prediction.created_at)).limit(1)
        pred_res = await db.execute(pred_stmt)
        pred = pred_res.scalar_one_or_none()

        rev_stmt = select(ClinicalReview).where(ClinicalReview.prediction_id == pred.id) if pred else None
        rev = None
        if rev_stmt:
            rev_res = await db.execute(rev_stmt)
            rev = rev_res.scalar_one_or_none()

        writer.writerow([
            enc.case_number,
            enc.encounter_date.isoformat(),
            enc.status,
            pred.predicted_class if pred else "",
            pred.predicted_label if pred else "",
            pred.confidence if pred else "",
            pred.uncertainty_level if pred else "",
            rev.decision if rev else "",
            rev.clinician_grade if rev else ""
        ])

    await log_audit_event(
        db=db,
        action="REPORT_EXPORTED_CSV",
        resource_type="screening",
        resource_id=None,
        user=current_user,
        details={"record_count": len(encounters), "format": "CSV"}
    )

    return Response(
        content=output.getvalue(),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=screening_records_export.csv"}
    )
