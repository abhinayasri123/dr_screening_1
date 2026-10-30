import json
import urllib.parse
from datetime import timedelta
from fastapi import APIRouter, Depends, HTTPException, status, Request
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from ..database import get_db
from ..models.user import User
from ..schemas.user import UserCreate, UserOut, TokenResponse, LoginRequest
from ..auth import (
    authenticate_user, 
    create_access_token, 
    get_password_hash, 
    get_current_active_user, 
    ACCESS_TOKEN_EXPIRE_MINUTES,
    log_audit_event
)

router = APIRouter(prefix="/auth", tags=["auth"])

@router.post("/login", response_model=TokenResponse)
async def login_for_access_token(
    request: Request,
    db: AsyncSession = Depends(get_db)
):
    username = None
    password = None

    content_type = request.headers.get("content-type", "")
    raw_body = await request.body()
    
    if "application/json" in content_type:
        try:
            body = json.loads(raw_body.decode("utf-8"))
            username = body.get("email") or body.get("username")
            password = body.get("password")
        except Exception:
            pass
    elif "application/x-www-form-urlencoded" in content_type:
        try:
            parsed = urllib.parse.parse_qs(raw_body.decode("utf-8"))
            username = (parsed.get("username") or parsed.get("email") or [None])[0]
            password = (parsed.get("password") or [None])[0]
        except Exception:
            pass
    else:
        # Fallback try json then form-urlencoded
        try:
            body = json.loads(raw_body.decode("utf-8"))
            username = body.get("email") or body.get("username")
            password = body.get("password")
        except Exception:
            try:
                parsed = urllib.parse.parse_qs(raw_body.decode("utf-8"))
                username = (parsed.get("username") or parsed.get("email") or [None])[0]
                password = (parsed.get("password") or [None])[0]
            except Exception:
                pass

    if not username or not password:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Missing username/email or password"
        )

    user = await authenticate_user(db, username, password)
    if not user:
        await log_audit_event(
            db=db,
            action="LOGIN_FAILED",
            resource_type="user",
            resource_id=None,
            user=None,
            outcome="FAILURE",
            details={"attempted_username": username, "reason": "Invalid credentials"}
        )
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    await log_audit_event(
        db=db,
        action="LOGIN_SUCCESS",
        resource_type="user",
        resource_id=user.id,
        user=user,
        outcome="SUCCESS",
        details={"email": user.email, "role": user.role}
    )

    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.email}, expires_delta=access_token_expires
    )
    return {
        "access_token": access_token, 
        "token_type": "bearer",
        "user": user
    }

@router.post("/register", response_model=UserOut)
async def register_user(
    user_in: UserCreate, 
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    if current_user.role != "admin":
        raise HTTPException(status_code=403, detail="Not authorized to create users")
        
    stmt = select(User).where(User.email == user_in.email)
    result = await db.execute(stmt)
    if result.scalar_one_or_none():
        raise HTTPException(status_code=400, detail="Email already registered")
        
    hashed_password = get_password_hash(user_in.password)
    db_user = User(
        email=user_in.email,
        hashed_password=hashed_password,
        full_name=user_in.full_name,
        role=user_in.role
    )
    db.add(db_user)
    await db.commit()
    await db.refresh(db_user)
    return db_user

@router.get("/me", response_model=UserOut)
async def read_users_me(current_user: User = Depends(get_current_active_user)):
    return current_user

@router.post("/logout")
async def logout():
    # Client side should discard the token
    return {"message": "Successfully logged out"}
