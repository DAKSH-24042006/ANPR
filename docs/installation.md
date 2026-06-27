# Installation Guide

## Prerequisites

| Requirement | Minimum Version |
| :--- | :--- |
| Python | 3.10+ |
| pip | 22.0+ |
| MySQL Server | 8.0+ |
| Git | 2.30+ |
| NVIDIA GPU (optional) | CUDA 11.8+ compatible |

---

## 1. Clone the Repository

```bash
git clone <repository-url>
cd ANPR
```

---

## 2. Create a Virtual Environment

```bash
python -m venv .venv
```

### Activate the Environment

**Windows (PowerShell):**
```powershell
.\.venv\Scripts\Activate.ps1
```

**Linux / macOS:**
```bash
source .venv/bin/activate
```

---

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### Required Packages

| Package | Purpose |
| :--- | :--- |
| `ultralytics>=8.3.0` | YOLO11 vehicle and plate detection |
| `torch>=2.0.0` | PyTorch deep learning backend |
| `torchvision>=0.15.0` | Image transforms and utilities |
| `opencv-python>=4.8.0` | Image loading, preprocessing, visualization |
| `pandas>=2.0.0` | Data manipulation for results analysis |
| `matplotlib>=3.7.0` | Plotting training curves |
| `pyyaml>=6.0` | YAML configuration parsing |
| `pillow>=10.0.0` | Image format support |
| `onnx>=1.14.0` | ONNX model export/inspection |
| `sqlalchemy>=2.0.0` | ORM database abstraction |
| `pymysql>=1.1.0` | MySQL connector |
| `python-multipart>=0.0.9` | Multipart file upload parsing |
| `fastapi>=0.110.0` | REST API framework |
| `uvicorn>=0.28.0` | ASGI server |
| `paddlepaddle` | PaddlePaddle framework (for OCR) |
| `paddlex` | PaddleOCR v5 engine |
| `psutil` | System resource monitoring (benchmarks) |

> **Note:** PaddlePaddle and PaddleX are required for the OCR engine but are
> not listed in `requirements.txt` because they have their own installation
> procedure. Install them following the official PaddlePaddle documentation
> for your platform and CUDA version.

---

## 4. Set Up MySQL Database

```sql
CREATE DATABASE IF NOT EXISTS anpr_db;
USE anpr_db;

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

Alternatively, execute the schema file directly:

```bash
mysql -u root -p < database/schema.sql
```

---

## 5. Configure Database Credentials

Edit `api/config.py` and set:

```python
DB_USER = "root"
DB_PASSWORD = "your_password"
DB_HOST = "localhost"
DB_PORT = 3306
DB_NAME = "anpr_db"
```

---

## 6. Verify Model Files

Ensure the following model files exist:

```text
models/plate_detector/best.pt    # Fine-tuned plate detector
yolo11s.pt                        # Pretrained vehicle detector (COCO)
```

---

## 7. Launch the Application

### CLI Inference
```bash
.venv/Scripts/python.exe run_inference.py <path_to_image>
```

### REST API Server
```bash
.venv/Scripts/python.exe -m uvicorn api.app:app --host 127.0.0.1 --port 8000
```

API documentation will be available at:
- Swagger UI: `http://127.0.0.1:8000/docs`
- ReDoc: `http://127.0.0.1:8000/redoc`

---

## 8. Run Tests

```bash
.venv/Scripts/python.exe -m pytest tests/system_tests.py -v
.venv/Scripts/python.exe -m pytest tests/api_tests.py -v
```

---

## 9. Run Benchmarks

```bash
.venv/Scripts/python.exe benchmarks/benchmark.py
```

Results will be saved to `benchmarks/benchmark_results.json` and `benchmarks/benchmark_report.md`.
