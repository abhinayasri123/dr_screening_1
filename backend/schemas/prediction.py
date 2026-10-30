from datetime import datetime
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

MEDICAL_DISCLAIMER = (
    "This system is an AI-assisted screening and decision-support tool. "
    "It is not a replacement for professional medical diagnosis. "
    "Final clinical decisions must be made by qualified healthcare professionals."
)

CLASS_NAMES = ["No DR", "Mild", "Moderate", "Severe", "Proliferative DR"]

class PredictionOut(BaseModel):
    id: str
    patient_id: str
    patient_name: Optional[str] = None
    patient_code: Optional[str] = None
    ehr_record_id: Optional[str] = None
    prediction_mode: str = "image_only"
    predicted_class: int
    predicted_label: str
    prediction: Optional[str] = None
    class_probabilities: List[float] = Field(default_factory=list)
    probabilities: Optional[Dict[str, float]] = None
    confidence: float
    uncertainty_std: float = 0.0
    uncertainty_level: str = "Low"
    model_version: str = "v1.0"
    is_demo_mode: bool = False
    review_status: str = "PENDING"
    image_path: Optional[str] = None
    image_filename: Optional[str] = None
    image_url: Optional[str] = None
    gradcam_url: Optional[str] = None
    gradcam_available: bool = False
    is_gradable: bool = True
    quality_status: str = "GRADABLE"
    quality_warnings: List[str] = Field(default_factory=list)
    clinical_recommendation: Optional[str] = None
    summary_status: Optional[str] = None  # "DR DETECTED" | "NO DR DETECTED" | "INCONCLUSIVE — CLINICIAN REVIEW REQUIRED"
    ood_detected: bool = False
    created_at: datetime
    created_by: str

    model_config = {"from_attributes": True}

class PredictionResponse(PredictionOut):
    disclaimer: str = MEDICAL_DISCLAIMER
