import uuid
from datetime import datetime, timezone
from sqlalchemy import Column, String, Integer, Float, Boolean, ForeignKey, DateTime
from sqlalchemy.orm import relationship

from ..database import Base

class EHRRecord(Base):
    """SQLAlchemy model for Electronic Health Records."""
    __tablename__ = "ehr_records"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    patient_id = Column(String(36), ForeignKey("patients.id"), nullable=False)
    
    age = Column(Integer, nullable=False)
    sex = Column(Integer, nullable=False)  # 0: Female, 1: Male
    diabetes_duration_years = Column(Float, nullable=False)
    hba1c = Column(Float, nullable=False)
    systolic_bp = Column(Float, nullable=False)
    diastolic_bp = Column(Float, nullable=False)
    bmi = Column(Float, nullable=False)
    
    smoking_status = Column(Integer, nullable=False)  # 0: Never, 1: Former, 2: Current
    insulin_use = Column(Boolean, nullable=False)
    prev_laser_treatment = Column(Boolean, nullable=False)
    nephropathy = Column(Boolean, nullable=False)
    neuropathy = Column(Boolean, nullable=False)
    cholesterol_ldl = Column(Float, nullable=False)
    
    is_synthetic = Column(Boolean, default=True)  # True for demo purposes
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

    # Relationships
    patient = relationship("Patient", back_populates="ehr_records")
    predictions = relationship("Prediction", back_populates="ehr_record")
