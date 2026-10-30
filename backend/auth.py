import os
from datetime import datetime, timedelta, timezone
from typing import Optional
try:
    from jose import jwt, JWTError
except ImportError:
    import jwt
    JWTError = jwt.PyJWTError
from passlib.context import CryptContext
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from pydantic import BaseModel

from .database import get_db
from .models.user import User

SECRET_KEY = os.getenv("SECRET_KEY", "super-secret-key-dr-screening-2026")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24  # 1 day

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/v1/auth/login")

class Token(BaseModel):
    access_token: str
    token_type: str

import hashlib
import secrets

def get_password_hash(password: str) -> str:
    salt = secrets.token_hex(16)
    hashed = hashlib.sha256((salt + password).encode('utf-8')).hexdigest()
    return f"{salt}${hashed}"

def verify_password(plain_password: str, hashed_password: str) -> bool:
    if "$" not in hashed_password:
        return False
    salt, hashed = hashed_password.split("$", 1)
    expected = hashlib.sha256((salt + plain_password).encode('utf-8')).hexdigest()
    return secrets.compare_digest(expected, hashed)

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """Create a new JWT access token."""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

async def get_current_user(token: str = Depends(oauth2_scheme), db: AsyncSession = Depends(get_db)) -> User:
    """Dependency to get the current authenticated user."""
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        email: str = payload.get("sub")
        if email is None:
            raise credentials_exception
    except (JWTError, Exception):
        raise credentials_exception

    stmt = select(User).where(User.email == email)
    result = await db.execute(stmt)
    user = result.scalar_one_or_none()
    
    if user is None:
        raise credentials_exception
    return user

async def get_current_active_user(current_user: User = Depends(get_current_user)) -> User:
    """Dependency to verify the current user is active."""
    if not current_user.is_active:
        raise HTTPException(status_code=400, detail="Inactive user")
    return current_user

async def require_admin(current_user: User = Depends(get_current_active_user)) -> User:
    if current_user.role != "admin":
        raise HTTPException(status_code=403, detail="Administrator role required for this action")
    return current_user

async def require_clinician(current_user: User = Depends(get_current_active_user)) -> User:
    if current_user.role not in ["clinician", "admin"]:
        raise HTTPException(status_code=403, detail="Clinician / Ophthalmologist authorization required")
    return current_user

async def require_technician(current_user: User = Depends(get_current_active_user)) -> User:
    if current_user.role not in ["technician", "clinician", "admin"]:
        raise HTTPException(status_code=403, detail="Screening technician authorization required")
    return current_user

async def require_researcher(current_user: User = Depends(get_current_active_user)) -> User:
    if current_user.role not in ["researcher", "clinician", "admin"]:
        raise HTTPException(status_code=403, detail="Researcher authorization required")
    return current_user

import json
import uuid
import hashlib
import secrets
from sqlalchemy import desc

def compute_audit_hash(
    prev_hash: Optional[str],
    event_id: str,
    timestamp_str: str,
    user_id: Optional[str],
    action: str,
    resource_type: str,
    resource_id: Optional[str],
    outcome: str,
    details: Optional[dict]
) -> str:
    """Computes deterministic SHA-256 hash for audit chain continuity."""
    canonical = {
        "prev_hash": prev_hash or ("0" * 64),
        "id": event_id,
        "timestamp": str(timestamp_str),
        "user_id": user_id or "",
        "action": action,
        "resource_type": resource_type,
        "resource_id": resource_id or "",
        "outcome": outcome or "SUCCESS",
        "details": details or {}
    }
    payload = json.dumps(canonical, sort_keys=True, separators=(',', ':'))
    return hashlib.sha256(payload.encode('utf-8')).hexdigest()

async def log_audit_event(
    db: AsyncSession,
    action: str,
    resource_type: str,
    resource_id: Optional[str] = None,
    user: Optional[User] = None,
    outcome: str = "SUCCESS",
    details: Optional[dict] = None,
    ip_address: Optional[str] = None
):
    """Asynchronously records an immutable audit event into audit_events table with cryptographic hash chaining."""
    try:
        from .models.audit import AuditEvent
        stmt = select(AuditEvent).order_by(desc(AuditEvent.sequence_num), desc(AuditEvent.created_at)).limit(1)
        res = await db.execute(stmt)
        last_event = res.scalar_one_or_none()

        next_seq = (last_event.sequence_num + 1) if (last_event and last_event.sequence_num is not None) else 1
        prev_hash = last_event.curr_hash if (last_event and last_event.curr_hash) else ("0" * 64)

        event_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc)
        ts_str = now.strftime("%Y-%m-%d %H:%M:%S.%f")

        curr_hash = compute_audit_hash(
            prev_hash=prev_hash,
            event_id=event_id,
            timestamp_str=ts_str,
            user_id=user.id if user else None,
            action=action,
            resource_type=resource_type,
            resource_id=resource_id,
            outcome=outcome,
            details=details
        )

        event = AuditEvent(
            id=event_id,
            sequence_num=next_seq,
            user_id=user.id if user else None,
            user_email=user.email if user else "SYSTEM",
            action=action,
            resource_type=resource_type,
            resource_id=resource_id,
            outcome=outcome,
            details=details or {},
            ip_address=ip_address,
            prev_hash=prev_hash,
            curr_hash=curr_hash,
            created_at=now
        )
        db.add(event)
        await db.commit()
        return event
    except Exception as e:
        print(f"[AuditLog] Failed to record audit event: {e}")

async def authenticate_user(db: AsyncSession, email: str, password: str) -> Optional[User]:
    """Authenticate a user by email and password."""
    stmt = select(User).where(User.email == email)
    result = await db.execute(stmt)
    user = result.scalar_one_or_none()
    
    if not user:
        return None
    if not verify_password(password, user.hashed_password):
        return None
    return user
