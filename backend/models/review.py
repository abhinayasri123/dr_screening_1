import uuid
from datetime import datetime, timezone
from sqlalchemy import Column, String, Integer, Boolean, ForeignKey, DateTime
from sqlalchemy.orm import relationship

from ..database import Base

class ClinicalReview(Base):
    """SQLAlchemy model for Clinical Reviews of Predictions."""
    __tablename__ = "clinical_reviews"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    prediction_id = Column(String(36), ForeignKey("predictions.id"), unique=True, nullable=False)
    reviewer_id = Column(String(36), ForeignKey("users.id"), nullable=False)
    
    agreed_with_ai = Column(Boolean, nullable=False)
    decision = Column(String, nullable=False, default="APPROVED")  # APPROVED, MODIFIED, REJECTED
    clinician_grade = Column(Integer, nullable=True)  # 0-4
    clinician_notes = Column(String, nullable=True)
    final_diagnosis = Column(String, nullable=False)
    
    requires_followup = Column(Boolean, nullable=False, default=False)
    followup_urgency = Column(String, nullable=True)  # routine/urgent/emergency
    
    review_timestamp = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

    # Relationships
    prediction = relationship("Prediction", back_populates="clinical_review")
    reviewer = relationship("User", backref="reviews")
