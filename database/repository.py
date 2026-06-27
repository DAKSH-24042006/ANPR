"""Database Repository Module for ANPR storage layer.

Abstracts SQLAlchemy CRUD queries (Inserts, paginated queries, deletes) to decouple database logic.
"""

import logging
from sqlalchemy.orm import Session
from database.models import Detection

logger = logging.getLogger("ANPRPipeline")

def add_detection(db: Session, detection: Detection) -> Detection:
    """Inserts a new ANPR detection record into the database.

    Args:
        db: SQLAlchemy database session.
        detection: SQLAlchemy model instance.

    Returns:
        The persisted detection instance, or None if database is offline/errored.
    """
    if db is None:
        logger.warning("Database session is offline. Skipping DB insert.")
        return None
        
    try:
        db.add(detection)
        db.commit()
        db.refresh(detection)
        logger.info(f"Database insert successful for UUID: {detection.uuid}")
        return detection
    except Exception as e:
        db.rollback()
        logger.error(f"Database transaction failed for UUID {detection.uuid}: {str(e)}")
        raise

def get_detections(db: Session, skip: int = 0, limit: int = 10) -> list:
    """Retrieves paginated list of detections ordered newest first.

    Args:
        db: SQLAlchemy database session.
        skip: Number of records to offset.
        limit: Max number of records to return.
    """
    if db is None:
        logger.warning("Database session is offline. Query returning empty history list.")
        return []
        
    try:
        return db.query(Detection).order_by(Detection.timestamp.desc()).offset(skip).limit(limit).all()
    except Exception as e:
        logger.error(f"Database query for history failed: {str(e)}")
        return []

def get_detection_by_uuid(db: Session, uuid_str: str) -> Detection:
    """Retrieves a single detection by UUID.

    Args:
        db: SQLAlchemy database session.
        uuid_str: The target record UUID.
    """
    if db is None:
        logger.warning("Database session is offline. Query returning None.")
        return None
        
    try:
        return db.query(Detection).filter(Detection.uuid == uuid_str).first()
    except Exception as e:
        logger.error(f"Database query for UUID {uuid_str} failed: {str(e)}")
        return None

def delete_detection_by_uuid(db: Session, uuid_str: str) -> bool:
    """Deletes a detection record from the database by UUID.

    Args:
        db: SQLAlchemy database session.
        uuid_str: The target record UUID to delete.

    Returns:
        True if record was found and deleted, False otherwise.
    """
    if db is None:
        logger.warning("Database session is offline. Delete query ignored.")
        return False
        
    try:
        record = db.query(Detection).filter(Detection.uuid == uuid_str).first()
        if record:
            db.delete(record)
            db.commit()
            logger.info(f"Database delete successful for UUID: {uuid_str}")
            return True
        logger.warning(f"Database record with UUID {uuid_str} not found for deletion.")
        return False
    except Exception as e:
        db.rollback()
        logger.error(f"Database delete transaction failed for UUID {uuid_str}: {str(e)}")
        raise


def update_detection(db: Session, uuid_str: str, update_data: dict) -> Detection:
    """Updates selected columns on a detection entry query by unique UUID.

    Args:
        db: SQLAlchemy database session.
        uuid_str: The target record UUID.
        update_data: Dictionary of column name keys and updated values.

    Returns:
        The updated database record, or None if connection is offline or not found.
    """
    if db is None:
        logger.warning("Database session is offline. Update query ignored.")
        return None
        
    try:
        record = db.query(Detection).filter(Detection.uuid == uuid_str).first()
        if record:
            for key, value in update_data.items():
                if hasattr(record, key):
                    setattr(record, key, value)
            db.commit()
            db.refresh(record)
            logger.info(f"Database update successful for UUID: {uuid_str}")
            return record
        logger.warning(f"Database record with UUID {uuid_str} not found for updates.")
        return None
    except Exception as e:
        db.rollback()
        logger.error(f"Database update transaction failed for UUID {uuid_str}: {str(e)}")
        raise


def get_detections_by_plate(db: Session, plate_num: str) -> list:
    """Retrieves all historical detections matching a license plate registration.

    Args:
        db: SQLAlchemy database session.
        plate_num: Target plate registration string.
    """
    if db is None:
        logger.warning("Database session is offline. Query returning empty plate list.")
        return []
        
    try:
        return db.query(Detection).filter(Detection.plate_number == plate_num.upper()).all()
    except Exception as e:
        logger.error(f"Database query by plate {plate_num} failed: {str(e)}")
        return []


def get_detections_by_timestamp(db: Session, start_time, end_time) -> list:
    """Retrieves all detections logged inside a timestamp range.

    Args:
        db: SQLAlchemy database session.
        start_time: Start datetime limit.
        end_time: End datetime limit.
    """
    if db is None:
        logger.warning("Database session is offline. Query returning empty range list.")
        return []
        
    try:
        return db.query(Detection).filter(
            Detection.timestamp >= start_time,
            Detection.timestamp <= end_time
        ).all()
    except Exception as e:
        logger.error(f"Database query by timestamp range failed: {str(e)}")
        return []

