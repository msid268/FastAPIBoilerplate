# app/db/session.py
from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker, Session
from contextlib import contextmanager
import logging
from urllib.parse import quote_plus
from app.core.config import settings

logger = logging.getLogger(__name__)

DATABASE_URL = (
    f"mssql+pyodbc://{settings.DATABASE_USER}:{quote_plus(settings.DATABASE_PASSWORD)}"
    f"@{settings.DATABASE_SERVER}/{settings.DATABASE_NAME}"
    f"?driver=ODBC+Driver+17+for+SQL+Server&TrustServerCertificate=yes&Encrypt=no"
)

engine = create_engine(
    DATABASE_URL,
    pool_size=settings.DB_POOL_SIZE,
    max_overflow=settings.DB_MAX_OVERFLOW,
    pool_timeout=settings.DB_POOL_TIMEOUT,
    pool_recycle=settings.DB_POOL_RECYCLE,
    pool_pre_ping=True,
    echo=settings.DB_ECHO,
    connect_args={"timeout": 30},
)

SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=True)

@event.listens_for(engine, "connect")
def receive_connect(dbapi_conn, connection_record):
    logger.debug("Database connection established.")

@event.listens_for(engine.pool, "checkin")
def receive_checkin(dbapi_conn, connection_record):
    logger.debug("Database connection returned to pool.")

def get_db():
    db = SessionLocal()
    try:
        logger.debug("Creating database session.")
        yield db
    except Exception as e:
        logger.error(f"Database session error: {e}", exc_info=True)
        db.rollback()
        raise
    finally:
        logger.debug("Closing database session.")
        db.close()

@contextmanager
def get_db_context():
    db = SessionLocal()
    try:
        logger.debug("Creating database context session.")
        yield db
        db.commit()
    except Exception as e:
        logger.error(f"Database context error: {e}", exc_info=True)
        db.rollback()
        raise
    finally:
        logger.debug("Closing database context session.")
        db.close()
