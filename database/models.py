"""Database models module for ANPR detections.

Defines the SQLAlchemy ORM schema mapping to the detections table.
"""

from datetime import datetime
from sqlalchemy import Column, Integer, String, Float, DateTime, Text
from database.database import Base

class Detection(Base):
    """ORM representation of the detections table."""
    __tablename__ = "detections"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    uuid = Column(String(36), unique=True, nullable=False, index=True)
    timestamp = Column(DateTime, nullable=False, default=datetime.now)
    
    # Vehicle and Plate Metadata
    vehicle_type = Column(String(50), nullable=True)
    plate_number = Column(String(20), nullable=True)
    vehicle_confidence = Column(Float, nullable=True)
    plate_confidence = Column(Float, nullable=True)
    ocr_confidence = Column(Float, nullable=True)
    
    # Timings & Files
    processing_time_ms = Column(Float, nullable=False)
    image_path = Column(String(500), nullable=False)
    annotated_image_path = Column(String(500), nullable=True)
    
    # Full prediction payload stored in JSON format text
    json_result = Column(Text, nullable=False)
