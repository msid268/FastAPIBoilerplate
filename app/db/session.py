"""
SQLAlchemy engine and session management.

What this module does
---------------------
This file is the single place where we:

- Build the SQLAlchemy **Engine** (connection to the database).
- Configure the **connection pool** (how connections are reused).
- Define how to create **Session** objects.
- Provide helpers to use sessions both in FastAPI dependencies and in
  plain Python code.

In other words: if your app needs to talk to the database, it goes
through something defined here.

Key concepts (high-level)
-------------------------
1. Engine
   - Represents the core interface to the database.
   - Knows *how* to connect (driver, user, password, host, DB name).
   - Manages the connection pool under the hood.

2. Session
   - A lightweight, short-lived wrapper around a DB connection.
   - Used for all ORM operations (query, add, delete, etc.).
   - Think of it as a "conversation" with the DB for a unit of work.

3. Connection Pool
   - Keeps a pool of open connections so we don't create a new TCP
     connection for every request.
   - Handles timeouts, recycling of old connections, and pre-ping
     (checking if a connection is alive before use).

4. Dependency vs Context Manager
   - `get_db` is a generator function meant to be used as a FastAPI
     dependency (`Depends(get_db)`).
   - `get_db_context` is a context manager for "regular" Python code
     (scripts, services, background tasks) where you want:

         with get_db_context() as db:
             ...

     and have commit/rollback handled for you.
"""

from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker, Session
from contextlib import contextmanager
import logging
from urllib.parse import quote_plus
from sqlalchemy.engine import URL
from app.core.config import settings

logger = logging.getLogger(__name__)

# Build the full database URL for SQL Server using pyodbc.
# We URL-encode the password to avoid issues with special characters.
# DATABASE_URL = (
#     f"mssql+pyodbc://{settings.DATABASE_USER}:{quote_plus(settings.DATABASE_PASSWORD)}"
#     f"@{settings.DATABASE_SERVER}/{settings.DATABASE_NAME}"
#     f"?driver=ODBC+Driver+17+for+SQL+Server&TrustServerCertificate=yes&Encrypt=no"
# )

DATABASE_URL = URL.create(
    drivername = "mssql+pyodbc",
    username = settings.DATABASE_USER,
    password= settings.DATABASE_PASSWORD,
    host = settings.DATABASE_SERVER,
    database = settings.DATABASE_NAME,
    query={
        "driver": "ODBC Driver 17 for SQL Server"
    }
)

# The Engine is the core object that knows how to talk to the database.
# Here we also configure connection pooling settings so we don't open a
# new connection on every request.
engine = create_engine(
    DATABASE_URL,
    pool_size=settings.DB_POOL_SIZE,           # how many connections to keep open
    max_overflow=settings.DB_MAX_OVERFLOW,     # extra connections allowed beyond pool_size
    pool_timeout=settings.DB_POOL_TIMEOUT,     # how long to wait for a connection from the pool
    pool_recycle=settings.DB_POOL_RECYCLE,     # recycle connections after N seconds (avoid stale)
    pool_pre_ping=True,                        # check connections are alive before using
    echo=settings.DB_ECHO,                     # log SQL queries if True
    connect_args={"timeout": 30},              # driver-level connect timeout (seconds)
)

# Session factory: each call to SessionLocal() will give you a new Session
# object bound to our engine. The Session itself is lightweight/cheap to create.
SessionLocal = sessionmaker(
    bind=engine,
    autoflush=False,        # don't autoflush on each query; commit/flush explicitly
    expire_on_commit=True,  # objects are expired after commit to avoid stale data
)

# --- Connection pool event listeners ----------------------------------------


@event.listens_for(engine, "connect")
def receive_connect(dbapi_conn, connection_record):
    """
    Hook that runs whenever a new DBAPI connection is created.

    This fires when the pool actually opens a new physical connection to the
    database (not every time you ask for a Session). We use it only for debug
    logging to understand pool behavior.
    """
    logger.debug("Database connection established.")


@event.listens_for(engine.pool, "checkin")
def receive_checkin(dbapi_conn, connection_record):
    """
    Hook that runs when a connection is returned to the pool.

    This does *not* mean the connection is closed, just that it's free to
    be reused by another request. Handy for debugging pool churn.
    """
    logger.debug("Database connection returned to pool.")


# --- Session helpers --------------------------------------------------------


def get_db():
    """
    FastAPI dependency that yields a database session.

    Usage (in a route):

        @router.get("/items")
        def read_items(db: Session = Depends(get_db)):
            return db.query(Item).all()

    Behaviour
    ---------
    - Creates a new SQLAlchemy Session for the duration of the request.
    - Yields it to the route handler.
    - On error, rolls back the transaction and re-raises.
    - On exit (success or failure), closes the session.

    Notes
    -----
    - This does *not* call `commit()` automatically. You should commit
      explicitly in your service layer or use `get_db_context` for
      one-off operations where an auto-commit pattern makes more sense.
    """
    db: Session = SessionLocal()
    try:
        logger.debug("Creating database session.")
        yield db
    except Exception as exc:
        logger.error("Database session error: %s", exc, exc_info=True)
        db.rollback()
        raise
    finally:
        logger.debug("Closing database session.")
        db.close()


@contextmanager
def get_db_context():
    """
    Context manager that provides a Session with automatic commit/rollback.

    Usage (outside FastAPI dependency system):

        from app.db.session import get_db_context

        def do_something():
            with get_db_context() as db:
                user = User(name="Alice")
                db.add(user)
                # No explicit commit needed here; it's done on successful exit.

    Behaviour
    ---------
    - Opens a new Session.
    - Yields it to the `with` block.
    - On normal exit: commits the transaction.
    - On exception: rolls back the transaction and re-raises.
    - Always closes the Session at the end.

    When to use which helper
    -------------------------
    - `get_db`:
      - Best for FastAPI routes where you want full control over
        commit/rollback in your service layer.
    - `get_db_context`:
      - Best for scripts, background jobs, or small "unit of work" helpers
        where "commit on success, rollback on error" is exactly what you want.
    """
    db: Session = SessionLocal()
    try:
        logger.debug("Creating database context session.")
        yield db
        db.commit()
    except Exception as exc:
        logger.error("Database context error: %s", exc, exc_info=True)
        db.rollback()
        raise
    finally:
        logger.debug("Closing database context session.")
        db.close()
