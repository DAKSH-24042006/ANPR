# Deployment Guide

## Development Environment

This project was developed and tested on the following environment:

### Hardware

| Component | Specification |
| :--- | :--- |
| CPU | Intel / AMD (64-bit) |
| RAM | 16 GB+ recommended |
| GPU | NVIDIA GPU with CUDA support (optional, accelerates YOLO inference) |
| Storage | ~2 GB for models + dataset |

### Software Stack

| Software | Version | Purpose |
| :--- | :--- | :--- |
| Windows | 10/11 (64-bit) | Operating System |
| Python | 3.13.x | Runtime |
| CUDA | 11.8+ (if GPU available) | GPU acceleration |
| PyTorch | 2.0+ | Deep learning backend |
| Ultralytics | 8.3+ | YOLO11 detection framework |
| PaddlePaddle | Latest stable | OCR framework |
| PaddleOCR/PaddleX | PP-OCRv5 | License plate text recognition |
| SQLAlchemy | 2.0+ | ORM database layer |
| MySQL | 8.0+ | Relational database |
| FastAPI | 0.110+ | REST API framework |
| Uvicorn | 0.28+ | ASGI server |
| OpenCV | 4.8+ | Image processing |

---

## Pre-Deployment Checklist

### 1. Environment Setup
- [ ] Python 3.10+ installed
- [ ] Virtual environment created (`.venv`)
- [ ] All packages installed from `requirements.txt`
- [ ] PaddlePaddle and PaddleX installed separately

### 2. Model Files
- [ ] `yolo11s.pt` exists in project root (pretrained vehicle detector)
- [ ] `models/plate_detector/best.pt` exists (fine-tuned plate detector)
- [ ] PaddleOCR model cache populated (auto-downloads on first run)

### 3. Database
- [ ] MySQL server running
- [ ] `anpr_db` database created
- [ ] `detections` table created (`database/schema.sql`)
- [ ] Credentials configured in `api/config.py`

### 4. Output Directories
- [ ] `outputs/` directory structure created (auto-created on startup)

### 5. Verification
- [ ] `run_inference.py` runs successfully on a test image
- [ ] FastAPI server starts (`uvicorn api.app:app`)
- [ ] `/health` endpoint returns `200 OK`
- [ ] `/detect` endpoint processes an image upload
- [ ] System tests pass (`pytest tests/system_tests.py`)

---

## Starting the Server

### Development Mode
```bash
.venv/Scripts/python.exe -m uvicorn api.app:app --host 127.0.0.1 --port 8000 --reload
```

### Production Mode
```bash
.venv/Scripts/python.exe -m uvicorn api.app:app --host 0.0.0.0 --port 8000 --workers 1
```

> **Note:** Due to model memory requirements, running multiple workers requires sufficient RAM. A single worker is recommended for CPU-only deployments.

---

## Logging

Application logs are written to:
```
outputs/logs/prediction.log
```

Logs include:
- Pipeline initialisation status
- Per-request UUID, timestamps, and processing times
- Detection results (vehicle type, plate number, confidences)
- Database operation outcomes
- Error and warning messages

---

## Troubleshooting

| Issue | Solution |
| :--- | :--- |
| MySQL connection refused | Verify MySQL is running and credentials in `api/config.py` are correct |
| Model files not found | Ensure `yolo11s.pt` and `models/plate_detector/best.pt` exist |
| PaddleOCR download fails | Set `PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True` and use cached models |
| High memory usage | Reduce image resolution or use GPU acceleration |
| Slow inference on CPU | OCR is the bottleneck; GPU acceleration recommended |
