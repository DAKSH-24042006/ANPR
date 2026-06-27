"""Pydantic validation schemas for ANPR FastAPI service.

Defines request/response models for REST API endpoints.
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime

# --- Detection Response Schemas ---

class VehicleDetails(BaseModel):
    type: Optional[str] = Field(None, example="motorcycle")
    bounding_box: List[int] = Field(default_factory=list, example=[62, 1, 255, 389])
    confidence: float = Field(0.0, example=0.882)

class PlateDetails(BaseModel):
    number: Optional[str] = Field(None, example="DL10SV4496")
    bounding_box: List[int] = Field(default_factory=list, example=[51, 3, 264, 393])
    confidence: float = Field(0.0, example=0.8934)
    ocr_confidence: float = Field(0.0, example=0.8976)

class TimingsMs(BaseModel):
    image_loading: float = 0.0
    preprocessing: float = 0.0
    vehicle_detection: float = 0.0
    vehicle_crop: float = 0.0
    plate_detection: float = 0.0
    plate_crop: float = 0.0
    image_enhancement: float = 0.0
    ocr: float = 0.0
    post_processing: float = 0.0
    image_save_ms: float = 0.0
    database_insert_ms: float = 0.0
    total: float = 0.0

class Metadata(BaseModel):
    vehicle_model: str = "YOLO11s COCO"
    plate_model: str = "YOLO11s Fine-Tuned"
    ocr_engine: str = "PP-OCRv5"
    processing_time_ms: float = 0.0
    image_width: int = 0
    image_height: int = 0
    timestamp: str

class DetectionResponse(BaseModel):
    uuid: Optional[str] = None
    image_path: Optional[str] = None
    annotated_image_path: Optional[str] = None
    status: str = Field(..., example="SUCCESS")
    message: Optional[str] = None
    vehicle: VehicleDetails
    plate: PlateDetails
    timings_ms: TimingsMs
    metadata: Metadata


# --- Database Record Response Schemas ---

class DetectionHistoryItem(BaseModel):
    id: int
    uuid: str
    timestamp: datetime
    vehicle_type: Optional[str] = None
    plate_number: Optional[str] = None
    vehicle_confidence: Optional[float] = None
    plate_confidence: Optional[float] = None
    ocr_confidence: Optional[float] = None
    processing_time_ms: float
    image_path: str
    annotated_image_path: Optional[str] = None
    json_result: Dict[str, Any]  # Parsed JSON string

    class Config:
        from_attributes = True


# --- Health Check Response ---

class HealthCheckResponse(BaseModel):
    status: str = "healthy"
    database: str = "connected"
    models: str = "loaded"
    # System information
    gpu_available: bool = False
    gpu_name: str = "N/A"
    cuda_version: str = "N/A"
    torch_version: str = "N/A"
    ultralytics_version: str = "N/A"
    paddleocr_version: str = "N/A"

