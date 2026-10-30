import uuid
from datetime import datetime, timezone
from sqlalchemy import Column, String, Integer, Float, Boolean, ForeignKey, DateTime, JSON
from ..database import Base

class FederatedExperiment(Base):
    """SQLAlchemy model for Federated Learning Simulations."""
    __tablename__ = "federated_experiments"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    name = Column(String, nullable=False)
    num_rounds = Column(Integer, nullable=False)
    num_clients = Column(Integer, nullable=False)
    strategy = Column(String, default="FedAvg", nullable=False)
    partition_type = Column(String, default="Non-IID", nullable=False) # 'IID' or 'Non-IID'
    local_epochs = Column(Integer, default=1, nullable=False)
    batch_size = Column(Integer, default=16, nullable=False)
    learning_rate = Column(Float, default=0.0001, nullable=False)
    
    global_accuracy = Column(Float, nullable=True)
    global_kappa = Column(Float, nullable=True)
    global_loss = Column(Float, nullable=True)
    status = Column(String, default="COMPLETED", nullable=False)
    rounds_history_json = Column(JSON, nullable=True)
    
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

class FederatedRound(Base):
    """SQLAlchemy model for Individual Communication Rounds."""
    __tablename__ = "federated_rounds"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    experiment_id = Column(String(36), ForeignKey("federated_experiments.id"), nullable=False)
    round_number = Column(Integer, nullable=False)
    loss = Column(Float, nullable=False)
    accuracy = Column(Float, nullable=False)
    kappa = Column(Float, nullable=False)
    participating_clients = Column(JSON, nullable=True)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

class FederatedClient(Base):
    """SQLAlchemy model for Simulated Hospital Client Nodes."""
    __tablename__ = "federated_clients"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    client_name = Column(String, nullable=False) # e.g. Hospital Client A, Hospital Client B, Hospital Client C
    hospital_name = Column(String, nullable=False)
    sample_count = Column(Integer, nullable=False)
    class_distribution_json = Column(JSON, nullable=True)
    last_round_loss = Column(Float, nullable=True)
    last_round_accuracy = Column(Float, nullable=True)
