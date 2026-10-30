import os
from typing import List, Optional, Dict, Any
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form, Query
from fastapi.responses import FileResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, desc

from ..database import get_db
from ..models.user import User
from ..models.patient import Patient
from ..models.prediction import Prediction
from ..schemas.prediction import PredictionResponse, PredictionOut, CLASS_NAMES
from ..auth import get_current_active_user
from ..utils.image_utils import save_upload_file, validate_image_file, ALLOWED_EXTENSIONS
from ..services.inference_service import get_inference_service

router = APIRouter(prefix="/predictions", tags=["predictions"])

from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
UPLOAD_DIR = os.path.abspath(PROJECT_ROOT / "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)


def _enrich_prediction(pred: Prediction, patient_code: Optional[str] = None, patient_name: Optional[str] = None) -> PredictionResponse:
    po = PredictionResponse.model_validate(pred)
    po.prediction = pred.predicted_label
    po.patient_code = patient_code
    po.patient_name = patient_name

    if pred.class_probabilities and len(pred.class_probabilities) == 5:
        po.probabilities = {
            CLASS_NAMES[i]: float(pred.class_probabilities[i]) for i in range(5)
        }

    if pred.image_path:
        fn = os.path.basename(pred.image_path)
        po.image_url = f"/api/v1/uploads/{fn}"
        if os.path.exists(pred.image_path):
            try:
                from ml.datasets.quality_checker import assess_image_quality
                q = assess_image_quality(pred.image_path)
                po.is_gradable = q.get("is_gradable", True)
                po.quality_status = q.get("quality_status", "GRADABLE")
                po.quality_warnings = q.get("reasons", [])
                po.clinical_recommendation = q.get("recommendation", "Standard Clinical Review")
            except Exception:
                pass

    # Calculate user-facing summary status
    if getattr(pred, "confidence", None) is not None:
        if not getattr(po, "is_gradable", True) or getattr(po, "ood_detected", False) or pred.confidence < 0.60 or pred.uncertainty_level == "High":
            po.summary_status = "INCONCLUSIVE — CLINICIAN REVIEW REQUIRED"
        elif pred.predicted_class > 0:
            po.summary_status = "DR DETECTED"
        else:
            po.summary_status = "NO DR DETECTED"

    if pred.gradcam_path and os.path.exists(pred.gradcam_path):
        gfn = os.path.basename(pred.gradcam_path)
        po.gradcam_url = f"/api/v1/uploads/{gfn}"
        po.gradcam_available = True
    else:
        po.gradcam_available = False
    return po


import json
from ..models.ehr_record import EHRRecord

@router.post("", response_model=PredictionResponse)
@router.post("/", response_model=PredictionResponse)
async def create_prediction(
    file: UploadFile = File(...),
    patient_id: Optional[str] = Form(None),
    ehr_record_id: Optional[str] = Form(None),
    ehr_data: Optional[str] = Form(None),
    mode: str = Form("image_only"),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    if not validate_image_file(file):
        raise HTTPException(status_code=400, detail=f"Invalid file type. Allowed: {ALLOWED_EXTENSIONS}")

    # Ensure valid patient_id
    patient = None
    if patient_id:
        stmt = select(Patient).where(Patient.id == patient_id)
        res = await db.execute(stmt)
        patient = res.scalar_one_or_none()

    if not patient:
        stmt = select(Patient).limit(1)
        res = await db.execute(stmt)
        patient = res.scalar_one_or_none()
        if not patient:
            import uuid, datetime
            patient = Patient(
                id=str(uuid.uuid4()),
                patient_code="P-1001",
                full_name="Default Patient",
                date_of_birth=datetime.date(1975, 1, 1),
                sex="M",
                created_by=current_user.id
            )
            db.add(patient)
            await db.commit()
            await db.refresh(patient)

    # Process EHR data for multimodal mode
    parsed_ehr = None
    if mode == "multimodal":
        if ehr_data:
            try:
                parsed_ehr = json.loads(ehr_data) if isinstance(ehr_data, str) else ehr_data
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Invalid EHR JSON data: {str(e)}")

        # If not passed directly in request, look up existing patient EHR
        if not parsed_ehr and patient:
            stmt = (
                select(EHRRecord)
                .where(EHRRecord.patient_id == patient.id)
                .order_by(desc(EHRRecord.created_at))
                .limit(1)
            )
            res = await db.execute(stmt)
            latest_ehr = res.scalar_one_or_none()
            if latest_ehr:
                ehr_record_id = latest_ehr.id
                parsed_ehr = {
                    "age": latest_ehr.age,
                    "sex": latest_ehr.sex,
                    "duration_diabetes_years": latest_ehr.diabetes_duration_years,
                    "hba1c": latest_ehr.hba1c,
                    "sbp": latest_ehr.systolic_bp,
                    "dbp": latest_ehr.diastolic_bp,
                    "bmi": latest_ehr.bmi,
                    "smoker": latest_ehr.smoking_status,
                    "insulin_dependent": latest_ehr.insulin_use,
                    "prev_laser_treatment": latest_ehr.prev_laser_treatment,
                    "nephropathy": latest_ehr.nephropathy,
                    "neuropathy": latest_ehr.neuropathy,
                    "cholesterol": latest_ehr.cholesterol_ldl,
                }

        # If still missing, return clear 422 validation error
        if not parsed_ehr:
            raise HTTPException(
                status_code=422,
                detail="Clinical EHR data is required for Multimodal Fusion mode. Please enter patient clinical factors or select a patient with recorded EHR."
            )

        # Validate EHR range & completeness
        from ml.preprocessing.ehr_preprocessor import validate_ehr_record
        is_valid, validation_errors = validate_ehr_record(parsed_ehr)
        if not is_valid:
            raise HTTPException(
                status_code=422,
                detail=f"Clinical EHR validation error: {'; '.join(validation_errors)}"
            )

        # Save or update EHR in database if newly provided
        if patient and ehr_data and not ehr_record_id:
            import datetime
            today = datetime.date.today()
            calc_age = today.year - patient.date_of_birth.year
            new_ehr_rec = EHRRecord(
                patient_id=patient.id,
                age=int(parsed_ehr.get("age", calc_age)),
                sex=1 if str(parsed_ehr.get("sex", patient.sex)).upper() in ["M", "1", "MALE"] else 0,
                diabetes_duration_years=float(parsed_ehr.get("diabetes_duration_years", parsed_ehr.get("duration_diabetes_years", 5.0))),
                hba1c=float(parsed_ehr.get("hba1c", 7.0)),
                systolic_bp=float(parsed_ehr.get("systolic_bp", parsed_ehr.get("sbp", 120.0))),
                diastolic_bp=float(parsed_ehr.get("diastolic_bp", parsed_ehr.get("dbp", 80.0))),
                bmi=float(parsed_ehr.get("bmi", 24.5)),
                smoking_status=int(parsed_ehr.get("smoking_status", 0)),
                insulin_use=bool(parsed_ehr.get("insulin_use", False)),
                prev_laser_treatment=bool(parsed_ehr.get("prev_laser_treatment", False)),
                nephropathy=bool(parsed_ehr.get("nephropathy", False)),
                neuropathy=bool(parsed_ehr.get("neuropathy", False)),
                cholesterol_ldl=float(parsed_ehr.get("cholesterol_ldl", parsed_ehr.get("cholesterol", 100.0))),
                is_synthetic=False
            )
            db.add(new_ehr_rec)
            await db.commit()
            await db.refresh(new_ehr_rec)
            ehr_record_id = new_ehr_rec.id

    # Save uploaded fundus image
    file_path = save_upload_file(file, UPLOAD_DIR)

    # 1. Optical and retinal authenticity screening (Stage A)
    from ml.datasets.quality_checker import assess_image_quality, EXACT_INVALID_IMAGE_MESSAGE
    quality = assess_image_quality(file_path)
    if not quality.get("is_retinal", True):
        # Remove rejected non-retinal image to prevent storage pollution
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
            except Exception:
                pass
        raise HTTPException(
            status_code=422,
            detail=EXACT_INVALID_IMAGE_MESSAGE
        )

    # Run real model inference
    inference_service = get_inference_service()
    try:
        result = inference_service.predict(file_path, ehr_dict=parsed_ehr, mode=mode)
    except (ValueError, FileNotFoundError) as ve:
        raise HTTPException(status_code=422, detail=str(ve))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Inference execution failed: {str(exc)}")

    # Generate Grad-CAM activation heatmap
    gradcam_filename = f"gradcam_{os.path.basename(file_path)}"
    gradcam_path = os.path.join(UPLOAD_DIR, gradcam_filename)
    try:
        inference_service.generate_gradcam(file_path, result.predicted_class, gradcam_path)
    except Exception as g_err:
        print(f"[Predictions] Grad-CAM generation note: {g_err}")
        gradcam_path = None

    # Save prediction to DB with PENDING review status
    db_pred = Prediction(
        patient_id=patient.id,
        ehr_record_id=ehr_record_id,
        image_path=file_path,
        image_filename=file.filename,
        prediction_mode=mode,
        predicted_class=result.predicted_class,
        predicted_label=result.predicted_label,
        class_probabilities=result.class_probabilities,
        confidence=result.confidence,
        uncertainty_std=result.uncertainty_std,
        uncertainty_level=result.uncertainty_level,
        gradcam_path=gradcam_path if os.path.exists(gradcam_path) else None,
        model_version=inference_service.model_version,
        is_demo_mode=result.is_demo_mode,
        review_status="PENDING",
        created_by=current_user.id
    )
    db.add(db_pred)

    # Synchronize with ScreeningEncounter & RetinalImage models so it immediately appears in Reports, Explainability, and Reviews
    from ..models.screening import ScreeningEncounter, RetinalImage, ClinicalMetadata
    from ..routers.screenings import generate_case_reference
    from ml.datasets.quality_checker import assess_image_quality

    quality = assess_image_quality(file_path)

    case_ref = generate_case_reference()
    encounter = ScreeningEncounter(
        case_number=case_ref,
        patient_id=patient.id,
        status="AWAITING_REVIEW",
        notes=f"Clinical DR Screening ({mode} mode)",
        created_by=current_user.id
    )
    db.add(encounter)
    await db.flush()
    db_pred.encounter_id = encounter.id

    retinal_img = RetinalImage(
        encounter_id=encounter.id,
        file_path=file_path,
        filename=file.filename or "fundus.jpg",
        width=quality.get("resolution", (300, 300))[0],
        height=quality.get("resolution", (300, 300))[1],
        blur_score=quality.get("blur_score", 120.0),
        mean_luminance=quality.get("mean_luminance", 120.0),
        is_gradable=quality.get("is_gradable", True),
        quality_status=quality.get("quality_status", "GRADABLE")
    )
    db.add(retinal_img)

    if parsed_ehr:
        try:
            meta = ClinicalMetadata(
                encounter_id=encounter.id,
                patient_id=patient.id,
                age=int(parsed_ehr.get("age", 50)) if parsed_ehr.get("age") is not None else None,
                diabetes_duration=float(parsed_ehr.get("diabetes_duration_years") or parsed_ehr.get("duration_diabetes_years", 5.0)),
                hba1c=float(parsed_ehr.get("hba1c", 7.0)),
                systolic_bp=float(parsed_ehr.get("systolic_bp") or parsed_ehr.get("sbp", 120.0)),
                diastolic_bp=float(parsed_ehr.get("diastolic_bp") or parsed_ehr.get("dbp", 80.0)),
                bmi=float(parsed_ehr.get("bmi", 24.5)) if parsed_ehr.get("bmi") is not None else None,
                creatinine=float(parsed_ehr.get("creatinine", 1.0)) if parsed_ehr.get("creatinine") is not None else None
            )
            db.add(meta)
        except Exception as meta_err:
            print(f"[Warning] Could not persist clinical metadata: {meta_err}")

    await db.commit()
    await db.refresh(db_pred)

    return _enrich_prediction(db_pred, patient_code=patient.patient_code, patient_name=patient.full_name)


@router.get("", response_model=List[PredictionResponse])
@router.get("/", response_model=List[PredictionResponse])
async def list_predictions(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=100),
    patient_id: Optional[str] = None,
    db: AsyncSession = Depends(get_db)
):
    stmt = select(Prediction, Patient.patient_code, Patient.full_name).outerjoin(
        Patient, Prediction.patient_id == Patient.id
    )
    if patient_id:
        stmt = stmt.where(Prediction.patient_id == patient_id)
    stmt = stmt.order_by(desc(Prediction.created_at)).offset(skip).limit(limit)

    result = await db.execute(stmt)
    rows = result.all()
    
    out_preds = []
    for pred, p_code, p_name in rows:
        out_preds.append(_enrich_prediction(pred, patient_code=p_code, patient_name=p_name))
    return out_preds


@router.get("/patient/{patient_id}", response_model=List[PredictionResponse])
async def get_predictions_by_patient(patient_id: str, db: AsyncSession = Depends(get_db)):
    stmt = select(Prediction, Patient.patient_code, Patient.full_name).outerjoin(
        Patient, Prediction.patient_id == Patient.id
    ).where(Prediction.patient_id == patient_id).order_by(desc(Prediction.created_at))

    result = await db.execute(stmt)
    rows = result.all()
    return [_enrich_prediction(p, patient_code=p_code, patient_name=p_name) for p, p_code, p_name in rows]


@router.get("/{id}", response_model=PredictionResponse)
async def get_prediction(id: str, db: AsyncSession = Depends(get_db)):
    stmt = select(Prediction, Patient.patient_code, Patient.full_name).outerjoin(
        Patient, Prediction.patient_id == Patient.id
    ).where(Prediction.id == id)
    result = await db.execute(stmt)
    row = result.first()
    if not row:
        raise HTTPException(status_code=404, detail="Prediction not found")
        
    pred, p_code, p_name = row
    return _enrich_prediction(pred, patient_code=p_code, patient_name=p_name)


@router.get("/{id}/gradcam")
async def get_gradcam(id: str, db: AsyncSession = Depends(get_db)):
    stmt = select(Prediction).where(Prediction.id == id)
    result = await db.execute(stmt)
    pred = result.scalar_one_or_none()
    if not pred or not pred.gradcam_path or not os.path.exists(pred.gradcam_path):
        raise HTTPException(status_code=404, detail="Grad-CAM visualization not found")
        
    return FileResponse(pred.gradcam_path, media_type="image/png")
