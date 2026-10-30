import os
from pathlib import Path
from dotenv import load_dotenv

env_path = Path(__file__).resolve().parent.parent / ".env"
if env_path.exists():
    load_dotenv(dotenv_path=env_path)
else:
    load_dotenv()
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from sqlalchemy.orm import declarative_base
from sqlalchemy.pool import NullPool

# Database URL from environment or default SQLite
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite+aiosqlite:///./dr_screening.db")

# In production (e.g., Render, Railway, Supabase), PostgreSQL URLs start with postgres:// or postgresql://
# SQLAlchemy async requires postgresql+asyncpg://
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql+asyncpg://", 1)
elif DATABASE_URL.startswith("postgresql://") and not DATABASE_URL.startswith("postgresql+asyncpg://"):
    DATABASE_URL = DATABASE_URL.replace("postgresql://", "postgresql+asyncpg://", 1)

# Create AsyncEngine with NullPool for SQLite to prevent cross-thread lock deadlocks
engine_kwargs = {"echo": False}
if "sqlite" in DATABASE_URL:
    engine_kwargs["poolclass"] = NullPool
    engine_kwargs["connect_args"] = {"check_same_thread": False}
else:
    engine_kwargs.update({
        "pool_size": 10,
        "max_overflow": 20,
        "pool_recycle": 1800,
        "pool_pre_ping": True
    })

engine = create_async_engine(DATABASE_URL, **engine_kwargs)

# Create session factory
async_sessionmaker_factory = async_sessionmaker(
    engine, class_=AsyncSession, expire_on_commit=False
)

# Declarative base for models
Base = declarative_base()

async def get_db():
    """
    Dependency to get the database session.
    Yields an AsyncSession.
    """
    async with async_sessionmaker_factory() as session:
        yield session

async def create_tables():
    """
    Create all tables in the database asynchronously or synchronously for SQLite.
    """
    if "sqlite" in DATABASE_URL:
        # Avoid aiosqlite thread deadlock during DDL schema creation on Windows SQLite
        sync_url = DATABASE_URL.replace("sqlite+aiosqlite://", "sqlite://")
        from sqlalchemy import create_engine
        sync_engine = create_engine(sync_url)
        Base.metadata.create_all(bind=sync_engine)
        sync_engine.dispose()
    else:
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
