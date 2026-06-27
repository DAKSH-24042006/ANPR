"""FastAPI Routes Module for ANPR application layer.

Registers all endpoint paths:
- POST /detect (runs pipeline, annotations, database save)
- GET /history (paginated query)
- GET /history/{id} (single item query by ID or UUID)
- DELETE /history/{id} (record and file deletes)
- GET /health (service checks)
"""

import json
import logging
from typing import List
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Query, status
from sqlalchemy.orm import Session

from api import schemas, config
from database.database import get_db, is_db_connected
from database.models import Detection
from database import repository
from services.detection_service import DetectionService

logger = logging.getLogger("ANPRPipeline")

router = APIRouter()

# Instantiate the shared detection service (caches loaded models once)
detection_service = DetectionService()


@router.get("/health", response_model=schemas.HealthCheckResponse)
def health_check(db: Session = Depends(get_db)):
    """Returns application, database connection, model cache health and system info."""
    db_status = "connected" if is_db_connected else "offline"
    models_status = "loaded" if detection_service.pipeline is not None else "failed"

    # ── GPU / CUDA information ────────────────────────────────────────────────
    try:
        import torch
        gpu_available = torch.cuda.is_available()
        gpu_name      = torch.cuda.get_device_name(0) if gpu_available else "N/A"
        cuda_version  = torch.version.cuda or "N/A"
        torch_ver     = torch.__version__
    except Exception:
        gpu_available = False
        gpu_name      = "N/A"
        cuda_version  = "N/A"
        torch_ver     = "N/A"

    try:
        import ultralytics
        ult_ver = ultralytics.__version__
    except Exception:
        ult_ver = "N/A"

    try:
        import paddleocr
        pocr_ver = getattr(paddleocr, "__version__", "installed")
    except Exception:
        pocr_ver = "N/A"

    return {
        "status":               "healthy",
        "database":             db_status,
        "models":               models_status,
        "gpu_available":        gpu_available,
        "gpu_name":             gpu_name,
        "cuda_version":         cuda_version,
        "torch_version":        torch_ver,
        "ultralytics_version":  ult_ver,
        "paddleocr_version":    pocr_ver,
    }



@router.post("/detect", response_model=schemas.DetectionResponse)
async def upload_and_detect(
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    """Processes uploaded image, draws overlays, and logs result records."""
    # 1. Size constraint check (10MB limit)
    contents = await file.read()
    file_size = len(contents)
    if file_size > config.MAX_UPLOAD_SIZE:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"Upload file exceeds maximum limit of {config.MAX_UPLOAD_SIZE / (1024 * 1024):.1f} MB."
        )

    # 2. Reset stream position (if needed, though we already read it)
    # 3. Call core detection service
    result = detection_service.process_image(contents, file.filename, db)
    
    # If the response indicates invalid file types, return with bad request status
    if result["status"] == "INVALID_IMAGE":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=result.get("message", "Invalid file extension.")
        )
        
    return result


@router.get("/history", response_model=List[schemas.DetectionHistoryItem])
def list_history(
    skip: int = Query(0, ge=0),
    limit: int = Query(10, ge=1, le=100),
    db: Session = Depends(get_db)
):
    """Retrieves paginated logs of prior detections ordered newest first."""
    db_records = repository.get_detections(db, skip=skip, limit=limit)
    
    # Parse json_result text strings back into python dictionaries
    results = []
    for record in db_records:
        try:
            parsed_json = json.loads(record.json_result)
        except Exception:
            parsed_json = {}
            
        results.append(
            schemas.DetectionHistoryItem(
                id=record.id,
                uuid=record.uuid,
                timestamp=record.timestamp,
                vehicle_type=record.vehicle_type,
                plate_number=record.plate_number,
                vehicle_confidence=record.vehicle_confidence,
                plate_confidence=record.plate_confidence,
                ocr_confidence=record.ocr_confidence,
                processing_time_ms=record.processing_time_ms,
                image_path=record.image_path,
                annotated_image_path=record.annotated_image_path,
                json_result=parsed_json
            )
        )
    return results


@router.get("/history/{id}", response_model=schemas.DetectionHistoryItem)
def get_history_item(id: str, db: Session = Depends(get_db)):
    """Retrieves a single detection log details query by numeric ID or string UUID."""
    if db is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Database connection offline. Record '{id}' not found."
        )
    record = None
    if id.isdigit():
        record = db.query(Detection).filter(Detection.id == int(id)).first()
    else:
        record = repository.get_detection_by_uuid(db, id)
        
    if not record:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Detection entry '{id}' not found."
        )
        
    try:
        parsed_json = json.loads(record.json_result)
    except Exception:
        parsed_json = {}
        
    return schemas.DetectionHistoryItem(
        id=record.id,
        uuid=record.uuid,
        timestamp=record.timestamp,
        vehicle_type=record.vehicle_type,
        plate_number=record.plate_number,
        vehicle_confidence=record.vehicle_confidence,
        plate_confidence=record.plate_confidence,
        ocr_confidence=record.ocr_confidence,
        processing_time_ms=record.processing_time_ms,
        image_path=record.image_path,
        annotated_image_path=record.annotated_image_path,
        json_result=parsed_json
    )


@router.delete("/history/{id}")
def delete_history_item(id: str, db: Session = Depends(get_db)):
    """Deletes detection entry and cleans up associated file assets."""
    if db is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Database connection offline. Record '{id}' not found for deletion."
        )
    target_uuid = id
    # If a numeric integer ID is supplied, resolve the UUID first
    if id.isdigit():
        record = db.query(Detection).filter(Detection.id == int(id)).first()
        if not record:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Detection entry '{id}' not found for deletion."
            )
        target_uuid = record.uuid
        
    success = detection_service.delete_record(target_uuid, db)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Detection entry '{id}' not found for deletion."
        )
        
    return {
        "status": "SUCCESS",
        "message": f"Successfully deleted record {id} and corresponding storage files."
    }
