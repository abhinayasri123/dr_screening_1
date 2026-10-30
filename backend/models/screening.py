import uuid
from datetime import datetime, timezone
from sqlalchemy import Column, String, Integer, Float, Boolean, ForeignKey, DateTime, JSON, Date
from sqlalchemy.orm import relationship
from ..database import Base

class ScreeningEncounter(Base):
    """SQLAlchemy model for Screening Encounters."""
    __tablename__ = "screening_encounters"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    case_number = Column(String, unique=True, index=True, nullable=False) # e.g. SCR-20261024-001
    patient_id = Column(String(36), ForeignKey("patients.id"), nullable=False)
    encounter_date = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    status = Column(String, default="AWAITING_IMAGE", nullable=False) # AWAITING_IMAGE, AWAITING_PREDICTION, AWAITING_REVIEW, APPROVED, MODIFIED, REJECTED, UNGRADABLE
    notes = Column(String, nullable=True)
    created_by = Column(String(36), ForeignKey("users.id"), nullable=False)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

    # Relationships
    patient = relationship("Patient", backref="encounters")
    retinal_images = relationship("RetinalImage", back_populates="encounter", cascade="all, delete-orphan")
    clinical_metadata = relationship("ClinicalMetadata", back_populates="encounter", uselist=False, cascade="all, delete-orphan")

class RetinalImage(Base):
    """SQLAlchemy model for Uploaded Retinal Fundus Images."""
    __tablename__ = "retinal_images"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    encounter_id = Column(String(36), ForeignKey("screening_encounters.id"), nullable=False)
    file_path = Column(String, nullable=False)
    filename = Column(String, nullable=False)
    width = Column(Integer, nullable=True)
    height = Column(Integer, nullable=True)
    blur_score = Column(Float, nullable=True)
    mean_luminance = Column(Float, nullable=True)
    is_gradable = Column(Boolean, default=True, nullable=False)
    quality_status = Column(String, default="GRADABLE", nullable=False)
    uploaded_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

    encounter = relationship("ScreeningEncounter", back_populates="retinal_images")

class ClinicalMetadata(Base):
    """SQLAlchemy model for Optional Patient Clinical Risk Factors."""
    __tablename__ = "clinical_metadata"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    encounter_id = Column(String(36), ForeignKey("screening_encounters.id"), nullable=True)
    patient_id = Column(String(36), ForeignKey("patients.id"), nullable=False)
    
    age = Column(Integer, nullable=True)
    diabetes_duration = Column(Float, nullable=True)
    hba1c = Column(Float, nullable=True)
    systolic_bp = Column(Float, nullable=True)
    diastolic_bp = Column(Float, nullable=True)
    bmi = Column(Float, nullable=True)
    creatinine = Column(Float, nullable=True)
    medical_history = Column(String, nullable=True)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

    encounter = relationship("ScreeningEncounter", back_populates="clinical_metadata")
