import uuid
from datetime import datetime, timezone
from sqlalchemy import Column, String, Integer, Float, Boolean, ForeignKey, DateTime, JSON
from sqlalchemy.orm import relationship

from ..database import Base

class Prediction(Base):
    """SQLAlchemy model for DR Predictions."""
    __tablename__ = "predictions"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    patient_id = Column(String(36), ForeignKey("patients.id"), nullable=False)
    encounter_id = Column(String(36), ForeignKey("screening_encounters.id"), nullable=True)
    ehr_record_id = Column(String(36), ForeignKey("ehr_records.id"), nullable=True)
    
    image_path = Column(String, nullable=True)
    image_filename = Column(String, nullable=True)
    prediction_mode = Column(String, nullable=False)  # image_only, ehr_only, multimodal
    
    predicted_class = Column(Integer, nullable=False)  # 0-4
    predicted_label = Column(String, nullable=False)
    class_probabilities = Column(JSON, nullable=False)
    confidence = Column(Float, nullable=False)
    
    uncertainty_std = Column(Float, nullable=False)
    uncertainty_level = Column(String, nullable=False)  # Low/Medium/High
    gradcam_path = Column(String, nullable=True)
    
    model_version = Column(String, nullable=False, default="v1.0")
    is_demo_mode = Column(Boolean, nullable=False, default=False)
    review_status = Column(String, nullable=False, default="PENDING")
    
    created_by = Column(String(36), ForeignKey("users.id"), nullable=False)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

    # Relationships
    patient = relationship("Patient", back_populates="predictions")
    encounter = relationship("ScreeningEncounter", backref="predictions")
    ehr_record = relationship("EHRRecord", back_populates="predictions")
    reviewer = relationship("User", backref="predictions")
    clinical_review = relationship("ClinicalReview", back_populates="prediction", uselist=False, cascade="all, delete-orphan")
