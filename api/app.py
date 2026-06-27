"""FastAPI Application entry point for ANPR service.

Registers the routes router, configures static mounts, and initializes database table
schemas upon startup via async lifespans.
"""

import os
# Force underlying C++ libraries (OpenMP, MKL, OpenBLAS) to use a single thread per request
# to prevent CPU context-switch thrashing and thread lock contention under FastAPI/Uvicorn.
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from api.routes import router
from database.database import Base, engine, is_db_connected

logger = logging.getLogger("ANPRPipeline")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handles startup schema checks and shutdown resource cleanups."""
    logger.info("FastAPI service starting up...")
    
    # Initialize SQLAlchemy database schema if MySQL is online
    if is_db_connected and engine is not None:
        try:
            Base.metadata.create_all(bind=engine)
            logger.info("MySQL database schemas verified and initialized.")
        except Exception as e:
            logger.error(f"Failed to verify database schemas: {str(e)}")
    else:
        logger.warning("Database offline. API running in degraded state.")
        
    yield
    
    logger.info("FastAPI service shutting down...")


# Initialize FastAPI app with descriptive metadata
app = FastAPI(
    title="ANPR API Service",
    description="REST API interface for Automatic Number Plate Recognition (Phase 3).",
    version="5.0.0",
    lifespan=lifespan
)

# Mount outputs directory as static to allow direct download of original/annotated image files
app.mount("/outputs", StaticFiles(directory="outputs"), name="outputs")

# Include central routing group
app.include_router(router)
