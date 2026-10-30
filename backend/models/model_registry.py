import uuid
from datetime import datetime, timezone
from sqlalchemy import Column, String, Float, Integer, Boolean, DateTime, JSON
from ..database import Base

class ModelVersion(Base):
    """SQLAlchemy model for Tracking & Registering Trained AI Models."""
    __tablename__ = "model_versions"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    name = Column(String, nullable=False, unique=True)
    architecture = Column(String, nullable=False, default="EfficientNet-B3")
    checkpoint_path = Column(String, nullable=False)
    checksum = Column(String, nullable=True)
    dataset_manifest = Column(String, nullable=True)
    
    accuracy = Column(Float, nullable=True)
    quadratic_weighted_kappa = Column(Float, nullable=True)
    macro_f1 = Column(Float, nullable=True)
    sensitivity = Column(Float, nullable=True)
    specificity = Column(Float, nullable=True)
    brier_score = Column(Float, nullable=True)
    metrics_json = Column(JSON, nullable=True)
    
    is_active = Column(Boolean, default=False, nullable=False)
    status = Column(String, default="VALIDATED", nullable=False) # 'ACTIVE', 'VALIDATED', 'ARCHIVED'
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    activated_at = Column(DateTime(timezone=True), nullable=True)

class TrainingRun(Base):
    """SQLAlchemy model for Tracking Centralized Model Training Runs."""
    __tablename__ = "training_runs"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    run_name = Column(String, nullable=False)
    model_architecture = Column(String, nullable=False)
    dataset_name = Column(String, nullable=False)
    epochs = Column(Integer, nullable=False)
    batch_size = Column(Integer, nullable=False)
    learning_rate = Column(Float, nullable=False)
    
    best_val_loss = Column(Float, nullable=True)
    best_val_kappa = Column(Float, nullable=True)
    best_val_accuracy = Column(Float, nullable=True)
    history_json = Column(JSON, nullable=True)
    status = Column(String, default="COMPLETED", nullable=False)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

class EvaluationRun(Base):
    """SQLAlchemy model for Held-out Test Set Evaluations."""
    __tablename__ = "evaluation_runs"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    model_version_id = Column(String(36), nullable=False)
    dataset_name = Column(String, nullable=False)
    test_sample_count = Column(Integer, nullable=False)
    
    accuracy = Column(Float, nullable=False)
    balanced_accuracy = Column(Float, nullable=False)
    macro_f1 = Column(Float, nullable=False)
    quadratic_weighted_kappa = Column(Float, nullable=False)
    sensitivity_referable = Column(Float, nullable=False)
    specificity_referable = Column(Float, nullable=False)
    brier_score = Column(Float, nullable=False)
    confusion_matrix_json = Column(JSON, nullable=False)
    per_class_metrics_json = Column(JSON, nullable=False)
    
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
