import os

from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

# Defaults to a SQLite file next to this script. Overridable via the
# DATABASE_URL env var -- e.g. in Docker, set to a path under the mounted
# /app/data volume so the database survives container restarts.
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./claims.db")

# check_same_thread=False is required for SQLite when used with FastAPI's
# threaded request handling.
engine = create_engine(
    DATABASE_URL, connect_args={"check_same_thread": False}
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()


def get_db():
    """FastAPI dependency that yields a DB session and always closes it."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()