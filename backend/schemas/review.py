from datetime import datetime
from typing import Optional
from pydantic import BaseModel

class ReviewBase(BaseModel):
    agreed_with_ai: bool = True
    decision: str = "APPROVED"  # APPROVED, MODIFIED, REJECTED
    clinician_grade: Optional[int] = None
    clinician_notes: Optional[str] = None
    final_diagnosis: Optional[str] = None
    requires_followup: bool = False
    followup_urgency: Optional[str] = "routine"

class ReviewCreate(ReviewBase):
    prediction_id: str

class ReviewUpdate(BaseModel):
    agreed_with_ai: Optional[bool] = None
    decision: Optional[str] = None
    clinician_grade: Optional[int] = None
    clinician_notes: Optional[str] = None
    final_diagnosis: Optional[str] = None
    requires_followup: Optional[bool] = None
    followup_urgency: Optional[str] = None

class ReviewOut(ReviewBase):
    id: str
    prediction_id: str
    reviewer_id: str
    review_timestamp: datetime

    model_config = {"from_attributes": True}
