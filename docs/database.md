# Database Documentation

## Overview

The ANPR system uses **MySQL 8.0+** with **SQLAlchemy 2.0 ORM** for database abstraction. All database operations are routed through `database/repository.py`, ensuring no raw SQL is executed in API routes or service layers.

---

## Schema

### Table: `detections`

```sql
CREATE TABLE IF NOT EXISTS detections (
    id INT AUTO_INCREMENT PRIMARY KEY,
    uuid VARCHAR(36) NOT NULL UNIQUE,
    timestamp DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    vehicle_type VARCHAR(50) DEFAULT NULL,
    plate_number VARCHAR(20) DEFAULT NULL,
    vehicle_confidence FLOAT DEFAULT NULL,
    plate_confidence FLOAT DEFAULT NULL,
    ocr_confidence FLOAT DEFAULT NULL,
    processing_time_ms FLOAT NOT NULL,
    image_path VARCHAR(500) NOT NULL,
    annotated_image_path VARCHAR(500) DEFAULT NULL,
    json_result TEXT NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
```

### Column Descriptions

| Column | Type | Nullable | Description |
| :--- | :--- | :---: | :--- |
| `id` | INT | No | Auto-incrementing primary key |
| `uuid` | VARCHAR(36) | No | UUIDv4 unique identifier (UNIQUE constraint) |
| `timestamp` | DATETIME | No | Detection timestamp (default: NOW) |
| `vehicle_type` | VARCHAR(50) | Yes | Detected vehicle class (car, motorcycle, bus, truck) |
| `plate_number` | VARCHAR(20) | Yes | Recognised license plate number |
| `vehicle_confidence` | FLOAT | Yes | YOLO vehicle detection confidence |
| `plate_confidence` | FLOAT | Yes | YOLO plate detection confidence |
| `ocr_confidence` | FLOAT | Yes | PP-OCRv5 recognition confidence |
| `processing_time_ms` | FLOAT | No | Total pipeline processing time in milliseconds |
| `image_path` | VARCHAR(500) | No | Relative path to original uploaded image |
| `annotated_image_path` | VARCHAR(500) | Yes | Relative path to annotated overlay image |
| `json_result` | TEXT | No | Complete JSON response payload |

---

## Repository Methods

All methods are defined in `database/repository.py`:

| Method | Description |
| :--- | :--- |
| `add_detection(db, record)` | Insert a new detection record |
| `get_detections(db, skip, limit)` | Paginated history query (newest first) |
| `get_detection_by_uuid(db, uuid_str)` | Fetch single record by UUID |
| `delete_detection_by_uuid(db, uuid_str)` | Delete record and return success boolean |
| `update_detection(db, uuid_str, data)` | Update specific columns by UUID |
| `get_detections_by_plate(db, plate_num)` | Search all records matching a plate number |
| `get_detections_by_timestamp(db, start, end)` | Query records within a datetime range |

---

## ORM Model

Defined in `database/models.py`:

```python
class Detection(Base):
    __tablename__ = "detections"

    id = Column(Integer, primary_key=True, autoincrement=True)
    uuid = Column(String(36), unique=True, nullable=False)
    timestamp = Column(DateTime, nullable=False)
    vehicle_type = Column(String(50))
    plate_number = Column(String(20))
    vehicle_confidence = Column(Float)
    plate_confidence = Column(Float)
    ocr_confidence = Column(Float)
    processing_time_ms = Column(Float, nullable=False)
    image_path = Column(String(500), nullable=False)
    annotated_image_path = Column(String(500))
    json_result = Column(Text, nullable=False)
```

---

## Connection Management

- **Engine:** Created in `database/database.py` using SQLAlchemy `create_engine` with `pool_pre_ping=True`.
- **Session Factory:** `SessionLocal` provides thread-safe sessions via `sessionmaker`.
- **Dependency Injection:** FastAPI routes receive sessions through `get_db()` generator with automatic cleanup.
- **Graceful Degradation:** If MySQL is unreachable on startup, the system continues in inference-only mode with file-based storage.

---

## Configuration

Database credentials are configured in `api/config.py`:

```python
DB_USER = "root"
DB_PASSWORD = "password"
DB_HOST = "localhost"
DB_PORT = 3306
DB_NAME = "anpr_db"
```

---

## Path Storage Convention

All file paths stored in the database are **relative paths** (e.g., `outputs/original_images/original_20260627_113235_uuid.jpg`). This ensures portability across different deployment environments.
