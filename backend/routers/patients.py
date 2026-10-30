import os
from typing import List, Optional
from datetime import date, datetime
from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, desc, func
from sqlalchemy.orm import selectinload

from ..database import get_db
from ..models.user import User
from ..models.patient import Patient
from ..models.ehr_record import EHRRecord
from ..models.prediction import Prediction
from ..schemas.patient import PatientCreate, PatientOut, PatientDetail, EHRRecordCreate, EHRRecordOut
from ..schemas.prediction import PredictionResponse, CLASS_NAMES
from ..auth import get_current_active_user

router = APIRouter(prefix="/patients", tags=["patients"])


def _enrich_patient_out(p: Patient, latest_pred: Optional[Prediction] = None, count: int = 0) -> PatientOut:
    today = date.today()
    age = today.year - p.date_of_birth.year - (
        (today.month, today.day) < (p.date_of_birth.month, p.date_of_birth.day)
    )
    po = PatientOut.model_validate(p)
    po.name = p.full_name
    po.age = age
    po.predictions_count = count
    if latest_pred:
        po.last_screening = latest_pred.created_at
        po.latest_ai_result = latest_pred.predicted_label
        po.review_status = latest_pred.review_status
    return po


@router.get("", response_model=List[PatientOut])
@router.get("/", response_model=List[PatientOut])
async def list_patients(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=100),
    db: AsyncSession = Depends(get_db)
):
    stmt = (
        select(Patient)
        .options(selectinload(Patient.predictions))
        .order_by(desc(Patient.created_at))
        .offset(skip)
        .limit(limit)
    )
    result = await db.execute(stmt)
    patients = result.scalars().all()

    out = []
    for p in patients:
        preds = p.predictions or []
        latest = sorted(preds, key=lambda x: x.created_at, reverse=True)[0] if preds else None
        out.append(_enrich_patient_out(p, latest, len(preds)))
    return out


@router.post("", response_model=PatientOut)
@router.post("/", response_model=PatientOut)
async def create_patient(
    patient_in: PatientCreate, 
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    stmt = select(Patient).where(Patient.patient_code == patient_in.patient_code)
    result = await db.execute(stmt)
    if result.scalar_one_or_none():
        raise HTTPException(status_code=400, detail="Patient code already registered")

    db_patient = Patient(
        **patient_in.model_dump(),
        created_by=current_user.id
    )
    db.add(db_patient)
    await db.commit()
    await db.refresh(db_patient)
    return _enrich_patient_out(db_patient, None, 0)


@router.get("/{id}", response_model=PatientDetail)
async def get_patient(id: str, db: AsyncSession = Depends(get_db)):
    stmt = (
        select(Patient)
        .options(
            selectinload(Patient.ehr_records),
            selectinload(Patient.predictions)
        )
        .where(Patient.id == id)
    )
    result = await db.execute(stmt)
    patient = result.scalar_one_or_none()
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")

    preds = sorted(patient.predictions or [], key=lambda x: x.created_at, reverse=True)
    latest = preds[0] if preds else None

    pd_out = PatientDetail.model_validate(patient)
    pd_out.name = patient.full_name
    today = date.today()
    pd_out.age = today.year - patient.date_of_birth.year
    pd_out.predictions_count = len(preds)
    if latest:
        pd_out.last_screening = latest.created_at
        pd_out.latest_ai_result = latest.predicted_label
        pd_out.review_status = latest.review_status

    enriched_preds = []
    for p in preds:
        po = PredictionResponse.model_validate(p)
        po.prediction = p.predicted_label
        po.patient_code = patient.patient_code
        po.patient_name = patient.full_name
        if p.class_probabilities and len(p.class_probabilities) == 5:
            po.probabilities = {
                CLASS_NAMES[i]: float(p.class_probabilities[i]) for i in range(5)
            }
        if p.image_path:
            po.image_url = f"/api/v1/uploads/{os.path.basename(p.image_path)}"
        if p.gradcam_path:
            po.gradcam_url = f"/api/v1/uploads/{os.path.basename(p.gradcam_path)}"
            po.gradcam_available = True
        enriched_preds.append(po)

    pd_out.predictions = enriched_preds
    return pd_out


@router.get("/{id}/predictions", response_model=List[PredictionResponse])
async def list_patient_predictions(id: str, db: AsyncSession = Depends(get_db)):
    stmt = select(Prediction, Patient.patient_code, Patient.full_name).outerjoin(
        Patient, Prediction.patient_id == Patient.id
    ).where(Prediction.patient_id == id).order_by(desc(Prediction.created_at))

    result = await db.execute(stmt)
    rows = result.all()

    enriched = []
    for p, p_code, p_name in rows:
        po = PredictionResponse.model_validate(p)
        po.prediction = p.predicted_label
        po.patient_code = p_code
        po.patient_name = p_name
        if p.class_probabilities and len(p.class_probabilities) == 5:
            po.probabilities = {
                CLASS_NAMES[i]: float(p.class_probabilities[i]) for i in range(5)
            }
        if p.image_path:
            po.image_url = f"/api/v1/uploads/{os.path.basename(p.image_path)}"
        if p.gradcam_path:
            po.gradcam_url = f"/api/v1/uploads/{os.path.basename(p.gradcam_path)}"
            po.gradcam_available = True
        enriched.append(po)
    return enriched


@router.get("/{id}/ehr", response_model=Optional[EHRRecordOut])
async def get_patient_ehr(id: str, db: AsyncSession = Depends(get_db)):
    stmt = (
        select(EHRRecord)
        .where(EHRRecord.patient_id == id)
        .order_by(desc(EHRRecord.created_at))
        .limit(1)
    )
    res = await db.execute(stmt)
    record = res.scalar_one_or_none()
    return record


@router.post("/{id}/ehr", response_model=EHRRecordOut)
async def create_or_update_patient_ehr(
    id: str,
    ehr_in: EHRRecordCreate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    stmt = select(Patient).where(Patient.id == id)
    res = await db.execute(stmt)
    patient = res.scalar_one_or_none()
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")

    hba1c_val = ehr_in.hba1c
    if hba1c_val is None and ehr_in.blood_glucose is not None:
        bg = ehr_in.blood_glucose
        if ehr_in.blood_glucose_unit == "mmol/L":
            bg = bg * 18.0182
        hba1c_val = round((bg + 46.7) / 28.7, 1)

    today = date.today()
    calculated_age = today.year - patient.date_of_birth.year - (
        (today.month, today.day) < (patient.date_of_birth.month, patient.date_of_birth.day)
    )
    rec_age = ehr_in.age if ehr_in.age is not None else calculated_age
    
    # Normalize sex
    if isinstance(ehr_in.sex, int):
        rec_sex = ehr_in.sex
    elif isinstance(ehr_in.sex, str):
        rec_sex = 1 if ehr_in.sex.upper() in ["M", "MALE", "1"] else 0
    else:
        rec_sex = 1 if patient.sex and patient.sex.upper() in ["M", "MALE", "1"] else 0

    # Normalize smoking status
    smoking_val = 0
    raw_smoke = ehr_in.smoking_status
    if isinstance(raw_smoke, int):
        smoking_val = raw_smoke
    elif isinstance(raw_smoke, str):
        smoking_val = 2 if "current" in raw_smoke.lower() else (1 if "former" in raw_smoke.lower() else 0)

    # Normalize insulin
    insulin_val = bool(ehr_in.insulin_use)
    if ehr_in.insulin_dependent is not None:
        insulin_val = ehr_in.insulin_dependent in [True, 1, "yes", "true", "True", "YES"]

    stmt = select(EHRRecord).where(EHRRecord.patient_id == id).order_by(desc(EHRRecord.created_at)).limit(1)
    res = await db.execute(stmt)
    existing_rec = res.scalar_one_or_none()

    if existing_rec:
        rec = existing_rec
        rec.age = rec_age
        rec.sex = rec_sex
        if ehr_in.diabetes_duration_years is not None:
            rec.diabetes_duration_years = ehr_in.diabetes_duration_years
        if hba1c_val is not None:
            rec.hba1c = hba1c_val
        if ehr_in.systolic_bp is not None:
            rec.systolic_bp = ehr_in.systolic_bp
        if ehr_in.diastolic_bp is not None:
            rec.diastolic_bp = ehr_in.diastolic_bp
        if ehr_in.bmi is not None:
            rec.bmi = ehr_in.bmi
        rec.smoking_status = smoking_val
        rec.insulin_use = insulin_val
        if ehr_in.prev_laser_treatment is not None:
            rec.prev_laser_treatment = ehr_in.prev_laser_treatment
        if ehr_in.nephropathy is not None:
            rec.nephropathy = ehr_in.nephropathy
        if ehr_in.neuropathy is not None:
            rec.neuropathy = ehr_in.neuropathy
        if ehr_in.cholesterol_ldl is not None:
            rec.cholesterol_ldl = ehr_in.cholesterol_ldl
        rec.is_synthetic = False
    else:
        rec = EHRRecord(
            patient_id=id,
            age=rec_age,
            sex=rec_sex,
            diabetes_duration_years=ehr_in.diabetes_duration_years if ehr_in.diabetes_duration_years is not None else 5.0,
            hba1c=hba1c_val if hba1c_val is not None else 7.0,
            systolic_bp=ehr_in.systolic_bp if ehr_in.systolic_bp is not None else 120.0,
            diastolic_bp=ehr_in.diastolic_bp if ehr_in.diastolic_bp is not None else 80.0,
            bmi=ehr_in.bmi if ehr_in.bmi is not None else 24.5,
            smoking_status=smoking_val,
            insulin_use=insulin_val,
            prev_laser_treatment=ehr_in.prev_laser_treatment or False,
            nephropathy=ehr_in.nephropathy or False,
            neuropathy=ehr_in.neuropathy or False,
            cholesterol_ldl=ehr_in.cholesterol_ldl if ehr_in.cholesterol_ldl is not None else 100.0,
            is_synthetic=False
        )
        db.add(rec)

    await db.commit()
    await db.refresh(rec)
    return rec


@router.delete("/{id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_patient(id: str, db: AsyncSession = Depends(get_db)):
    stmt = select(Patient).where(Patient.id == id)
    result = await db.execute(stmt)
    patient = result.scalar_one_or_none()
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")
        
    await db.delete(patient)
    await db.commit()
