"""Detection Service Module for ANPR application layer.

Orchestrates directory creations, unique UUID requests, file saves (raw, annotated, json),
inference pipeline execution, database persistence, and deletion file cleanups.
"""

import os
import uuid
import json
import logging
from datetime import datetime
from pathlib import Path
from sqlalchemy.orm import Session

from src import config as src_config
from api import config as api_config
from src.anpr_pipeline import ANPRPipeline
from src.visualization import draw_detections
from src.response_builder import build_response
from database.models import Detection
from database import repository

logger = logging.getLogger("ANPRPipeline")

class DetectionService:
    """Orchestrator service handling business logic for ANPR inference requests."""
    
    def __init__(self):
        """Initializes and caches the underlying ANPR pipeline."""
        # Warm up the pipeline once on startup to cache models
        device = getattr(src_config, "DEVICE", "cpu")
        self.pipeline = ANPRPipeline(device=device)
        self._ensure_output_directories()

    def _ensure_output_directories(self):
        """Creates output folders if they do not exist on disk."""
        folders = [
            "outputs/original_images",
            "outputs/annotated_images",
            "outputs/json_results",
            "outputs/logs"
        ]
        for folder in folders:
            Path(folder).mkdir(parents=True, exist_ok=True)

    def process_image(self, file_content: bytes, filename: str, db: Session = None) -> dict:
        """Executes ANPR pipeline, saves files, and stores database entry.

        Args:
            file_content: Raw image byte content from request.
            filename: Original file name.
            db: SQLAlchemy session (if database is online).

        Returns:
            Standardized API response dictionary.
        """
        import time
        request_uuid = str(uuid.uuid4())
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 1. Validate extension
        ext = os.path.splitext(filename)[1].lower()
        if ext not in api_config.ALLOWED_EXTENSIONS:
            return build_response(
                status="INVALID_IMAGE",
                message=f"Unsupported file format '{ext}'. Allowed formats: {api_config.ALLOWED_EXTENSIONS}",
                uuid=request_uuid
            )
            
        # 2. Write original image to disk (Measure raw save time)
        orig_filename = f"original_{timestamp_str}_{request_uuid}{ext}"
        orig_relative_path = f"outputs/original_images/{orig_filename}"
        orig_absolute_path = Path(orig_relative_path)
        
        t_start_raw = time.perf_counter()
        try:
            with open(orig_absolute_path, "wb") as f:
                f.write(file_content)
        except Exception as e:
            logger.error(f"Failed to save original uploaded image to disk: {str(e)}")
            return build_response(
                status="INVALID_IMAGE",
                message="Failed to write uploaded image to storage.",
                uuid=request_uuid
            )
        t_end_raw = time.perf_counter()
        raw_save_time = (t_end_raw - t_start_raw) * 1000.0

        # 3. Execute ANPR Inference Pipeline
        pipeline_result = None
        try:
            pipeline_result = self.pipeline.run(str(orig_absolute_path))
        except Exception as e:
            logger.error(f"Inference pipeline execution crashed for UUID {request_uuid}: {str(e)}")
            # Cleanup saved original image
            if orig_absolute_path.exists():
                orig_absolute_path.unlink()
            return build_response(
                status="INVALID_IMAGE",
                message=f"AI pipeline processing failed: {str(e)}",
                uuid=request_uuid,
                image_path=orig_relative_path
            )

        # 4. Determine status and construct standardized response base
        temp_resp = build_response(pipeline_result)
        status = temp_resp["status"]

        # 5. Conditional overlays & save annotated image (Measure overlay save time)
        annotated_relative_path = None
        annotated_save_time = 0.0
        if status in ["SUCCESS", "NO_PLATE", "OCR_FAILED"]:
            try:
                t_start_ann = time.perf_counter()
                annotated_relative_path = draw_detections(
                    str(orig_absolute_path),
                    pipeline_result,
                    request_uuid
                )
                t_end_ann = time.perf_counter()
                annotated_save_time = (t_end_ann - t_start_ann) * 1000.0
            except Exception as e:
                logger.error(f"Failed to create annotated image overlay: {str(e)}")

        # 6. Build response object with UUID and image path attributes
        response = build_response(
            pipeline_result=pipeline_result,
            uuid=request_uuid,
            image_path=orig_relative_path,
            annotated_image_path=annotated_relative_path
        )

        # 7. Save JSON result (Measure JSON save time)
        json_filename = f"json_{timestamp_str}_{request_uuid}.json"
        json_relative_path = f"outputs/json_results/{json_filename}"
        t_start_json = time.perf_counter()
        try:
            with open(Path(json_relative_path), "w", encoding="utf-8") as f:
                json.dump(response, f, indent=4)
        except Exception as e:
            logger.error(f"Failed to save json output file to disk: {str(e)}")
        t_end_json = time.perf_counter()
        json_save_time = (t_end_json - t_start_json) * 1000.0

        # Calculate total image saving speed (raw + annotated + JSON write times)
        image_save_ms = raw_save_time + annotated_save_time + json_save_time
        response["timings_ms"]["image_save_ms"] = float(round(image_save_ms, 2))

        # 8. Database Persistence (Measure database insert transaction time)
        database_insert_ms = 0.0
        if db is not None:
            try:
                # Map variables securely
                vehicle = response.get("vehicle")
                plate = response.get("plate")
                timings = response.get("timings_ms", {})
                metadata = response.get("metadata", {})
                
                # Format SQL-safe values
                veh_type = vehicle.get("type") if vehicle else None
                plate_num = plate.get("number") if plate else None
                veh_conf = vehicle.get("confidence") if vehicle else None
                plate_conf = plate.get("confidence") if plate else None
                ocr_conf = plate.get("ocr_confidence") if plate else None
                
                # Parse timestamp for models
                time_now = datetime.now()
                if metadata.get("timestamp"):
                    try:
                        time_now = datetime.strptime(metadata["timestamp"], "%Y-%m-%d %H:%M:%S")
                    except ValueError:
                        pass
                
                detection_record = Detection(
                    uuid=request_uuid,
                    timestamp=time_now,
                    vehicle_type=veh_type,
                    plate_number=plate_num,
                    vehicle_confidence=veh_conf,
                    plate_confidence=plate_conf,
                    ocr_confidence=ocr_conf,
                    processing_time_ms=timings.get("total", 0.0),
                    image_path=orig_relative_path,
                    annotated_image_path=annotated_relative_path,
                    json_result=json.dumps(response)
                )
                
                t_start_db = time.perf_counter()
                repository.add_detection(db, detection_record)
                t_end_db = time.perf_counter()
                database_insert_ms = (t_end_db - t_start_db) * 1000.0
            except Exception as e:
                logger.error(f"Database save bypassed due to connection failure: {str(e)}")

        # Inject final database latency timing measurement
        response["timings_ms"]["database_insert_ms"] = float(round(database_insert_ms, 2))

        # Return standardized response dictionary
        return response

    def delete_record(self, request_uuid: str, db: Session) -> bool:
        """Deletes database detection entry and removes associated disk files.

        Args:
            request_uuid: The target detection UUID.
            db: SQLAlchemy session.

        Returns:
            True if deleted successfully, False if not found.
        """
        # 1. Fetch database entry
        record = repository.get_detection_by_uuid(db, request_uuid)
        if not record:
            logger.warning(f"Detection record {request_uuid} not found for file deletion.")
            return False

        # 2. Extract file paths
        orig_path = Path(record.image_path)
        annotated_path = Path(record.annotated_image_path) if record.annotated_image_path else None
        
        # Deduce json filename using the same timestamp + uuid pattern
        base_name = orig_path.stem.replace("original_", "")
        json_path = Path(f"outputs/json_results/json_{base_name}.json")

        # 3. Delete database record
        repository.delete_detection_by_uuid(db, request_uuid)

        # 4. Remove physical files from disk (suppress errors if already missing)
        for filepath in [orig_path, annotated_path, json_path]:
            if filepath and filepath.exists():
                try:
                    filepath.unlink()
                    logger.info(f"Successfully deleted output file: {filepath}")
                except Exception as e:
                    logger.error(f"Could not delete physical file {filepath}: {str(e)}")

        return True
