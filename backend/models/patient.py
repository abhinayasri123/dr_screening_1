import uuid
from datetime import datetime, timezone
from sqlalchemy import Column, String, Date, ForeignKey, DateTime
from sqlalchemy.orm import relationship

from ..database import Base

class Patient(Base):
    """SQLAlchemy model for Patients."""
    __tablename__ = "patients"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    patient_code = Column(String, unique=True, index=True, nullable=False)
    full_name = Column(String, nullable=False)
    date_of_birth = Column(Date, nullable=False)
    sex = Column(String, nullable=False)  # M/F/Other
    contact_email = Column(String, nullable=True)
    
    created_by = Column(String(36), ForeignKey("users.id"), nullable=False)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))

    # Relationships
    user = relationship("User", backref="patients")
    ehr_records = relationship("EHRRecord", back_populates="patient", cascade="all, delete-orphan")
    predictions = relationship("Prediction", back_populates="patient", cascade="all, delete-orphan")
