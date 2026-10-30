import uuid
from datetime import datetime, timezone
from sqlalchemy import Column, String, DateTime, JSON, Integer
from ..database import Base

class AuditEvent(Base):
    """SQLAlchemy model for Clinical & System Immutable Audit Events with Cryptographic Hash Chaining."""
    __tablename__ = "audit_events"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    sequence_num = Column(Integer, nullable=True, index=True)
    user_id = Column(String(36), nullable=True)
    user_email = Column(String, nullable=True)
    action = Column(String, nullable=False, index=True)  # e.g., 'LOGIN', 'PREDICTION_GENERATED', 'CLINICIAN_REVIEW_SAVED', 'MODEL_ACTIVATED'
    resource_type = Column(String, nullable=False)       # 'screening', 'prediction', 'review', 'model', 'user', 'audit'
    resource_id = Column(String, nullable=True)
    outcome = Column(String(20), nullable=False, default="SUCCESS", index=True) # 'SUCCESS', 'FAILURE'
    details = Column(JSON, nullable=True)
    ip_address = Column(String, nullable=True)
    prev_hash = Column(String(64), nullable=True)
    curr_hash = Column(String(64), nullable=True, index=True)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), index=True)
