# REST API Documentation

## Base URL

```
http://127.0.0.1:8000
```

## Interactive Documentation

| Path | Description |
| :--- | :--- |
| `/docs` | Swagger UI (interactive) |
| `/redoc` | ReDoc (read-only) |

---

## Endpoints

### GET /health

Returns system health status including database connectivity and model availability.

**Response:**
```json
{
    "status": "healthy",
    "database": "connected",
    "models": "loaded"
}
```

| Field | Type | Description |
| :--- | :--- | :--- |
| `status` | string | Always `"healthy"` if the server is responding |
| `database` | string | `"connected"` or `"offline"` |
| `models` | string | `"loaded"` or `"failed"` |

---

### POST /detect

Accepts a multipart image upload, runs the complete ANPR pipeline, and returns a structured detection result.

**Request:**
- Content-Type: `multipart/form-data`
- Field: `file` (required) — JPEG or PNG image file

**Example (cURL):**
```bash
curl -X POST http://127.0.0.1:8000/detect \
  -F "file=@test_image.jpg"
```

**Example (Python):**
```python
import requests

with open("test_image.jpg", "rb") as f:
    r = requests.post("http://127.0.0.1:8000/detect", files={"file": f})
print(r.json())
```

**Success Response (200):**
```json
{
    "uuid": "0dbff119-0ad8-4aa7-aa59-e8c954c78336",
    "image_path": "outputs/original_images/original_20260627_113235_0dbff119.jpg",
    "annotated_image_path": "outputs/annotated_images/annotated_20260627_113243_0dbff119.jpg",
    "status": "SUCCESS",
    "vehicle": {
        "type": "motorcycle",
        "bounding_box": [62, 1, 255, 389],
        "confidence": 0.8821
    },
    "plate": {
        "number": "DL10SV4496",
        "bounding_box": [51, 3, 264, 393],
        "confidence": 0.8935,
        "ocr_confidence": 0.8976
    },
    "timings_ms": {
        "image_loading": 28.88,
        "preprocessing": 2.5,
        "vehicle_detection": 2352.55,
        "vehicle_crop": 0.54,
        "plate_detection": 354.67,
        "plate_crop": 0.29,
        "image_enhancement": 6.55,
        "ocr": 4777.57,
        "post_processing": 0.78,
        "image_save_ms": 35.82,
        "database_insert_ms": 0.0,
        "total": 7524.38
    },
    "metadata": {
        "vehicle_model": "YOLO11s-COCO",
        "plate_model": "YOLO11s-FineTuned",
        "ocr_engine": "PaddleOCR-v5",
        "processing_time_ms": 7524.38,
        "image_width": 320,
        "image_height": 426,
        "timestamp": "2026-06-27 11:32:43"
    }
}
```

**Possible Status Values:**

| Status | Description |
| :--- | :--- |
| `SUCCESS` | Vehicle detected, plate read successfully |
| `NO_VEHICLE` | No supported vehicle found in image |
| `NO_PLATE` | Vehicle found but no plate detected |
| `OCR_FAILED` | Plate detected but OCR returned empty or below threshold |
| `INVALID_IMAGE` | Image could not be loaded, is corrupted, or has unsupported format |

---

### GET /history

Returns paginated detection history, newest first.

**Query Parameters:**

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `page` | int | 1 | Page number |
| `limit` | int | 10 | Records per page |

**Example:**
```bash
curl http://127.0.0.1:8000/history?page=1&limit=5
```

---

### GET /history/{id}

Returns a single detection record by numeric ID or UUID string.

**Path Parameters:**

| Parameter | Type | Description |
| :--- | :--- | :--- |
| `id` | string | Numeric database ID or UUID string |

**Example:**
```bash
curl http://127.0.0.1:8000/history/1
curl http://127.0.0.1:8000/history/0dbff119-0ad8-4aa7-aa59-e8c954c78336
```

**404 Response (not found):**
```json
{"detail": "Detection record '999' not found."}
```

---

### DELETE /history/{id}

Deletes a detection record and all associated files (original image, annotated image, JSON result).

**Path Parameters:**

| Parameter | Type | Description |
| :--- | :--- | :--- |
| `id` | string | Numeric database ID or UUID string |

**Success Response (200):**
```json
{"detail": "Detection record deleted successfully."}
```

**404 Response:**
```json
{"detail": "Detection record '999' not found for deletion."}
```

---

## Error Handling

All endpoints return structured JSON error responses. The system degrades gracefully when the database is offline — inference continues and files are saved locally, but database persistence and history queries are unavailable.

| HTTP Code | Meaning |
| :--- | :--- |
| 200 | Success |
| 404 | Record not found |
| 422 | Validation error (missing file, wrong format) |
| 500 | Internal server error |
