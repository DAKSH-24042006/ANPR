"""Database Module for ANPR storage layer.

Sets up the SQLAlchemy engine, session factory, declarative base, and checks
MySQL database connectivity, with graceful fallback/error tracking.
"""

import logging
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, declarative_base
from api import config

logger = logging.getLogger("ANPRPipeline")

DATABASE_URL = f"mysql+pymysql://{config.DB_USER}:{config.DB_PASSWORD}@{config.DB_HOST}:{config.DB_PORT}/{config.DB_NAME}"

Base = declarative_base()
engine = None
SessionLocal = None
is_db_connected = False

try:
    # 1. Create SQLAlchemy engine for MySQL
    engine = create_engine(
        DATABASE_URL,
        pool_pre_ping=True,
        pool_recycle=3600,
        connect_args={"connect_timeout": 5}
    )
    
    # 2. Test database connection
    with engine.connect() as conn:
        conn.execute(text("SELECT 1"))
    
    is_db_connected = True
    # 3. Create thread-local session factory
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    logger.info("Successfully connected to MySQL database.")
    
except Exception as e:
    logger.error(
        f"MySQL database connection failed: {str(e)}. "
        "Running in degraded mode (Inference only, database storage disabled)."
    )
    is_db_connected = False
    SessionLocal = None
    engine = None

def get_db():
    """Dependency generator to retrieve database sessions.

    Yields:
        SQLAlchemy session instance, or None if connection is offline.
    """
    if not is_db_connected or SessionLocal is None:
        yield None
        return
        
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
