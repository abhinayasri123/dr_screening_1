from datetime import date, datetime
from typing import Optional, List, Any
from pydantic import BaseModel, EmailStr, Field

class EHRRecordBase(BaseModel):
    age: Optional[int] = None
    sex: Optional[Any] = Field(default=0, description="0: Female, 1: Male")
    diabetes_duration_years: Optional[float] = None
    hba1c: Optional[float] = None
    blood_glucose: Optional[float] = None
    blood_glucose_unit: Optional[str] = "mg/dL"
    systolic_bp: Optional[float] = None
    diastolic_bp: Optional[float] = None
    bmi: Optional[float] = None
    smoking_status: Optional[Any] = Field(default=0, description="0: Never, 1: Former, 2: Current")
    insulin_use: Optional[Any] = False
    insulin_dependent: Optional[Any] = None
    prev_laser_treatment: Optional[bool] = False
    nephropathy: Optional[bool] = False
    neuropathy: Optional[bool] = False
    cholesterol_ldl: Optional[float] = None
    is_synthetic: bool = False

class EHRRecordCreate(EHRRecordBase):
    pass

class EHRRecordOut(EHRRecordBase):
    id: str
    patient_id: str
    created_at: datetime

    model_config = {"from_attributes": True}

class PatientBase(BaseModel):
    patient_code: str
    full_name: str
    date_of_birth: date
    sex: str
    contact_email: Optional[EmailStr] = None

class PatientCreate(PatientBase):
    pass

class PatientOut(PatientBase):
    id: str
    name: Optional[str] = None
    age: Optional[int] = None
    last_screening: Optional[datetime] = None
    latest_ai_result: Optional[str] = None
    review_status: Optional[str] = None
    predictions_count: int = 0
    created_by: str
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}

class PatientDetail(PatientOut):
    ehr_records: List[EHRRecordOut] = []
    predictions: List[Any] = []

    model_config = {"from_attributes": True}
