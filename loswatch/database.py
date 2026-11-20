"""SQLite setup. Every connection enforces foreign keys."""
import os
from pathlib import Path

from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker

from loswatch.models import Base


def make_engine(url: str | None = None):
    if url is None:
        url = os.environ.get("LOSWATCH_DATABASE_URL")
    if url is None:
        directory = Path(__file__).resolve().parent.parent / "data"
        directory.mkdir(exist_ok=True)
        url = f"sqlite:///{(directory / 'loswatch.db').as_posix()}"
    if not url.startswith("sqlite:"):
        raise ValueError("Phase 1 supports SQLite only.")
    engine = create_engine(url, connect_args={"check_same_thread": False, "timeout": 15})

    @event.listens_for(engine, "connect")
    def configure_sqlite(connection, _):
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute("PRAGMA busy_timeout=15000")

    return engine


def initialize(engine):
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine, expire_on_commit=False)
