# ANPR System Architecture

## Overview

The Automatic Number Plate Recognition (ANPR) system is built as a **seven-layer architecture** where each layer has a single, well-defined responsibility. Layers are loosely coupled and communicate through standardised Python dictionaries.

---

## Layer Diagram

```text
┌─────────────────────────────────────────────────┐
│  L1 – Image Acquisition & Preprocessing         │
│       image_loader.py  ·  preprocessing.py       │
├─────────────────────────────────────────────────┤
│  L2 – Vehicle Detection & Cropping               │
│       vehicle_detector.py  ·  vehicle_crop.py     │
├─────────────────────────────────────────────────┤
│  L3 – Plate Detection & Cropping                 │
│       plate_detector.py  ·  plate_crop.py         │
├─────────────────────────────────────────────────┤
│  L4 – Image Enhancement                          │
│       image_enhancement.py                        │
├─────────────────────────────────────────────────┤
│  L5 – OCR & Post-Processing                      │
│       ocr_engine.py  ·  post_processing.py        │
├─────────────────────────────────────────────────┤
│  L6 – Result Presentation & Application           │
│       visualization.py  ·  response_builder.py    │
├─────────────────────────────────────────────────┤
│  L7 – Storage & Integration                       │
│       database/  ·  services/  ·  api/            │
└─────────────────────────────────────────────────┘
```

---

## Layer Descriptions

### Layer 1 – Image Acquisition & Preprocessing
- **Modules:** `src/image_loader.py`, `src/preprocessing.py`
- **Responsibility:** Load images from disk, validate dimensions, convert colour spaces, and apply preprocessing (resize, normalise).

### Layer 2 – Vehicle Detection & Cropping
- **Modules:** `src/vehicle_detector.py`, `src/vehicle_crop.py`
- **Model:** YOLO11s pretrained on COCO
- **Supported Classes:** Car, Motorcycle, Bus, Truck
- **Responsibility:** Detect vehicles in the preprocessed image and crop the region of interest with configurable padding.

### Layer 3 – Plate Detection & Cropping
- **Modules:** `src/plate_detector.py`, `src/plate_crop.py`
- **Model:** YOLO11s fine-tuned on a custom Indian license plate dataset
- **Responsibility:** Detect license plates within the vehicle crop and extract the plate region.

### Layer 4 – Image Enhancement
- **Module:** `src/image_enhancement.py`
- **Techniques:** CLAHE contrast enhancement, optional perspective correction, optional denoising
- **Responsibility:** Improve plate crop quality before OCR.

### Layer 5 – OCR & Post-Processing
- **Modules:** `src/ocr_engine.py`, `src/post_processing.py`
- **Engine:** PP-OCRv5 (PaddlePaddle)
- **Responsibility:** Extract raw text from the enhanced plate image, clean the result, and validate against Indian plate format patterns.

### Layer 6 – Result Presentation & Application
- **Modules:** `src/visualization.py`, `src/response_builder.py`
- **Responsibility:** Draw annotated bounding boxes on the original image, build standardised JSON response payloads with UUID, image paths, and timing metrics.

### Layer 7 – Storage & Integration
- **Modules:** `database/`, `services/detection_service.py`, `api/`
- **Responsibility:** Persist results to MySQL via SQLAlchemy ORM, orchestrate the end-to-end flow through `DetectionService`, and expose REST endpoints via FastAPI.

---

## Execution Flow

```text
Input Image
    ↓
Image Loading  (L1)
    ↓
Preprocessing  (L1)
    ↓
Vehicle Detection  (L2)
    ↓
Vehicle Crop  (L2)
    ↓
Plate Detection  (L3)
    ↓
Plate Crop  (L3)
    ↓
Image Enhancement  (L4)
    ↓
OCR Text Extraction  (L5)
    ↓
Post-Processing  (L5)
    ↓
Visualization  (L6)
    ↓
Response Builder  (L6)
    ↓
Detection Service  (L7)
    ↓
Database Repository  (L7)
    ↓
API Response  (L7)
```

---

## Pipeline Orchestrator

The `src/anpr_pipeline.py` module (`ANPRPipeline` class) orchestrates Layers 1–5, measuring latency for each stage with `PerformanceTimer`.

The `services/detection_service.py` module (`DetectionService` class) orchestrates Layers 6–7, calling the pipeline, saving files, building responses, and persisting to the database.

---

## Project Structure

```text
ANPR/
├── api/
│   ├── app.py              # FastAPI application setup
│   ├── config.py            # API & database configuration
│   ├── routes.py            # REST endpoint definitions
│   └── schemas.py           # Pydantic response models
├── benchmarks/
│   ├── benchmark.py         # Performance benchmarking script
│   ├── benchmark_report.md  # Generated benchmark report
│   └── benchmark_results.json
├── database/
│   ├── database.py          # SQLAlchemy engine & session factory
│   ├── models.py            # ORM model definitions
│   ├── repository.py        # CRUD repository functions
│   └── schema.sql           # MySQL DDL
├── docs/                    # Project documentation
├── models/
│   └── plate_detector/
│       ├── best.pt          # Fine-tuned plate detector weights
│       └── last.pt          # Last checkpoint
├── outputs/
│   ├── original_images/     # Uploaded images
│   ├── annotated_images/    # Overlay visualizations
│   ├── json_results/        # JSON response files
│   └── logs/                # Application logs
├── services/
│   └── detection_service.py # Business logic orchestration
├── src/
│   ├── anpr_pipeline.py     # Pipeline orchestrator (L1-L5)
│   ├── config.py            # Inference configuration
│   ├── image_loader.py      # Image loading (L1)
│   ├── preprocessing.py     # Image preprocessing (L1)
│   ├── vehicle_detector.py  # YOLO vehicle detection (L2)
│   ├── vehicle_crop.py      # Vehicle ROI cropping (L2)
│   ├── plate_detector.py    # YOLO plate detection (L3)
│   ├── plate_crop.py        # Plate ROI cropping (L3)
│   ├── image_enhancement.py # Plate enhancement (L4)
│   ├── ocr_engine.py        # PP-OCRv5 engine (L5)
│   ├── post_processing.py   # Text cleanup (L5)
│   ├── response_builder.py  # JSON response builder (L6)
│   ├── visualization.py     # Bounding box overlay (L6)
│   └── utils.py             # Shared utilities
├── tests/
│   ├── system_tests.py      # Pipeline integration tests
│   └── api_tests.py         # REST API tests
├── training/
│   └── train.py             # Model training script
├── run_inference.py          # CLI entry point
├── requirements.txt
└── README.md
```

---

## Design Principles

- **SOLID:** Single responsibility per module; dependency injection via function parameters.
- **Loose Coupling:** Layers communicate through plain Python dictionaries; no layer directly imports another layer's internals.
- **Graceful Degradation:** If MySQL is offline, the system continues inference-only with file storage.
- **Configuration Driven:** All thresholds, paths, and parameters are centralised in `src/config.py` (inference) and `api/config.py` (application).
