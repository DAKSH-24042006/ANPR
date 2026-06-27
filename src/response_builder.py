"""Response Builder Module for ANPR Inference Pipeline.

Structures end-to-end pipeline results into standardized, schema-compliant JSON payloads.
Handles SUCCESS, NO_VEHICLE, NO_PLATE, OCR_FAILED, and INVALID_IMAGE scenarios.
"""

from datetime import datetime

def build_response(
    pipeline_result: dict = None,
    status: str = "SUCCESS",
    message: str = None,
    uuid: str = None,
    image_path: str = None,
    annotated_image_path: str = None
) -> dict:
    """Builds a schema-compliant response payload.

    Args:
        pipeline_result: Raw dict output from ANPRPipeline.run().
        status: Operation outcome status.
        message: Optional diagnostic string.
        uuid: Unique request identifier.
        image_path: Relative path to original uploaded image.
        annotated_image_path: Relative path to saved overlay image.

    Returns:
        Structured response dictionary.
    """
    # 1. Initialize default/empty sub-components
    vehicle_out = {
        "type": None,
        "bounding_box": [],
        "confidence": 0.0
    }
    
    plate_out = {
        "number": None,
        "bounding_box": [],
        "confidence": 0.0,
        "ocr_confidence": 0.0
    }
    
    timings_out = {
        "image_loading": 0.0,
        "preprocessing": 0.0,
        "vehicle_detection": 0.0,
        "vehicle_crop": 0.0,
        "plate_detection": 0.0,
        "plate_crop": 0.0,
        "image_enhancement": 0.0,
        "ocr": 0.0,
        "post_processing": 0.0,
        "total": 0.0
    }
    
    metadata_out = {
        "vehicle_model": "YOLO11s COCO",
        "plate_model": "YOLO11s Fine-Tuned",
        "ocr_engine": "PP-OCRv5",
        "processing_time_ms": 0.0,
        "image_width": 0,
        "image_height": 0,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

    # 2. Handle invalid image input or general crashes
    if status == "INVALID_IMAGE" or pipeline_result is None:
        return {
            "uuid": uuid,
            "image_path": image_path,
            "annotated_image_path": annotated_image_path,
            "status": "INVALID_IMAGE",
            "message": message or "Invalid, corrupted, or unsupported image file.",
            "vehicle": vehicle_out,
            "plate": plate_out,
            "timings_ms": timings_out,
            "metadata": metadata_out
        }

    # 3. Extract metrics and timings from pipeline execution
    timings_raw = pipeline_result.get("timings_ms", {})
    timings_out = {
        "image_loading": timings_raw.get("image_loading", 0.0),
        "preprocessing": timings_raw.get("preprocessing", 0.0),
        "vehicle_detection": timings_raw.get("vehicle_detection", 0.0),
        "vehicle_crop": timings_raw.get("vehicle_cropping", 0.0),
        "plate_detection": timings_raw.get("plate_detection", 0.0),
        "plate_crop": timings_raw.get("plate_cropping", 0.0),
        "image_enhancement": timings_raw.get("image_enhancement", 0.0),
        "ocr": timings_raw.get("ocr", 0.0),
        "post_processing": timings_raw.get("post_processing", 0.0),
        "total": timings_raw.get("total_inference", 0.0)
    }

    metadata_raw = pipeline_result.get("metadata", {})
    metadata_out = {
        "vehicle_model": metadata_raw.get("vehicle_model", "YOLO11s COCO"),
        "plate_model": metadata_raw.get("plate_model", "YOLO11s Fine-Tuned"),
        "ocr_engine": metadata_raw.get("ocr_engine", "PP-OCRv5"),
        "processing_time_ms": metadata_raw.get("processing_time_ms", 0.0),
        "image_width": metadata_raw.get("image_width", 0),
        "image_height": metadata_raw.get("image_height", 0),
        "timestamp": metadata_raw.get("timestamp", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    }

    # 4. Map Vehicle Details
    vehicle_raw = pipeline_result.get("vehicle")
    if vehicle_raw:
        vehicle_out = {
            "type": vehicle_raw.get("type"),
            "bounding_box": vehicle_raw.get("box", []),
            "confidence": vehicle_raw.get("confidence", 0.0)
        }
    else:
        # No vehicle detected
        return {
            "uuid": uuid,
            "image_path": image_path,
            "annotated_image_path": annotated_image_path,
            "status": "NO_VEHICLE",
            "message": message or "No supported vehicle detected.",
            "vehicle": vehicle_out,
            "plate": plate_out,
            "timings_ms": timings_out,
            "metadata": metadata_out
        }

    # 5. Map Plate Details
    plate_raw = pipeline_result.get("plate")
    if plate_raw and plate_raw.get("box") is not None:
        plate_out = {
            "number": plate_raw.get("clean_text") or None,
            "bounding_box": plate_raw.get("box", []),
            "confidence": plate_raw.get("confidence", 0.0),
            "ocr_confidence": plate_raw.get("ocr_confidence", 0.0)
        }
        
        # OCR Validation Check
        if not plate_raw.get("raw_text"):
            status_ret = "OCR_FAILED"
            msg_ret = "OCR text recognition failed or returned empty."
        elif not plate_out["number"]:
            status_ret = "OCR_FAILED"
            msg_ret = "License plate number rejected due to low OCR confidence."
        else:
            status_ret = "SUCCESS"
            msg_ret = None
    else:
        # Vehicle found but no plate detected
        return {
            "uuid": uuid,
            "image_path": image_path,
            "annotated_image_path": annotated_image_path,
            "status": "NO_PLATE",
            "message": message or "License plate not detected.",
            "vehicle": vehicle_out,
            "plate": plate_out,
            "timings_ms": timings_out,
            "metadata": metadata_out
        }

    return {
        "uuid": uuid,
        "image_path": image_path,
        "annotated_image_path": annotated_image_path,
        "status": status_ret,
        **({"message": msg_ret} if msg_ret else {}),
        "vehicle": vehicle_out,
        "plate": plate_out,
        "timings_ms": timings_out,
        "metadata": metadata_out
    }
