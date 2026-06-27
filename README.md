# Automatic Number Plate Recognition (ANPR) System - Phase 1

This repository contains **Phase 1 (Training Pipeline)** of a production-quality Automatic Number Plate Recognition (ANPR) system. This phase handles training a license plate detector by fine-tuning **YOLO11s** on an Indian License Plate Dataset, validating the dataset structure, evaluating the model, and exporting it to ONNX/TensorRT for downstream inference.

---

## Project Structure

The project structure is organized as follows:

```
ANPR/
│
├── .venv/                   # Python virtual environment (ignored by Git)
│
├── datasets/                # Windows Directory Junction mapping to dataset
│   ├── images/              # Image files split into train, valid, test
│   │   ├── train/
│   │   ├── valid/
│   │   └── test/
│   ├── labels/              # Bounding box labels (YOLO format)
│   │   ├── train/
│   │   ├── valid/
│   │   └── test/
│   └── data.yaml            # Original Roboflow YOLO dataset configuration
│
├── training/
│   ├── config.py            # Global parameter and path configuration
│   ├── train.py             # Dataset verification & YOLO11s fine-tuning
│   ├── evaluate.py          # Model verification and performance metric reporter
│   ├── predict.py           # Sample inference visualizer
│   ├── export.py            # Model exporter (ONNX & TensorRT)
│   └── requirements.txt     # Script dependencies
│
├── models/
│   └── plate_detector/      # Destination folder for trained weights (best.pt, last.pt)
│
├── outputs/                 # Directory for training curves, logs, and prediction visualizations
│
├── requirements.txt         # Project-level dependencies
├── README.md                # System documentation
└── .gitignore               # git exclude rules
```

---

## Hardware & GPU Requirements

The pipeline is optimized and configured for the following development machine:
- **CPU:** Intel Core i5-13420H
- **GPU:** NVIDIA GeForce RTX 4050 Laptop GPU (6GB VRAM)
- **RAM:** 16GB
- **OS:** Windows 11
- **PyTorch Version:** $\ge 2.0.0$ (CUDA-compiled)
- **CUDA Toolkit Version:** Recommended 11.8 or 12.x to match PyTorch wheels

### Optimizations Implemented:
1. **Automatic Mixed Precision (AMP):** Enabled during training to speed up calculations and reduce VRAM footprint.
2. **Optimal Batch Size (16):** Calibrated to maximize processing speed on 6GB VRAM while avoiding Out-Of-Memory (OOM) errors.
3. **Multiprocessing Workers (0):** Configured to `0` on Windows inside `train.py` to prevent multi-processing overhead and hangs typical of PyTorch/Ultralytics on Windows.

---

## Setup & Installation

### 1. Environment Creation
Create a Python 3.11 virtual environment in the root of the project:
```powershell
py -3.11 -m venv .venv
```

### 2. Dependency Installation
Activate the virtual environment and install the required libraries:
```powershell
# Activate venv
.\.venv\Scripts\Activate.ps1

# Install requirements
pip install -r requirements.txt
```

### 3. Dataset Placement
The pipeline expects the dataset to be in YOLO format under `datasets/`.
You can map your dataset directory to `datasets/` by creating a Windows Directory Junction:
```powershell
# Open cmd/powershell as Administrator and run:
cmd /c mklink /j d:\ANPR\datasets "C:\Users\daksh\Downloads\ANPR_Final_Dataset_Split"
```

---

## Pipeline Execution

Ensure your virtual environment is active before running any script.

### 1. Dataset Validation & Model Training
To run the bidirectional validation and execute transfer learning:
```powershell
python training/train.py
```
- **What it does:**
  - Loads configurations from `config.py`.
  - Performs **Bidirectional Dataset Validation** (checking both Image $\rightarrow$ Label and Label $\rightarrow$ Image paths) and prints a validation report checking for corrupt images, missing labels, missing images, invalid annotations, and orphan label files.
  - Automatically prints CUDA availability, GPU name, PyTorch and Python versions.
  - Prints class distribution counts and image resolution range statistics.
  - Saves a training configuration backup (`training_config.yaml`) inside `outputs/`.
  - Starts training for `100` epochs with early stopping (`patience=20`).
  - Automatically identifies where YOLO saved the output using the official Ultralytics API (`results.save_dir`), then copies `best.pt`, `last.pt`, `results.csv`, `results.png`, curves, and matrices to their permanent output directories.
  - Logs the process inside `outputs/logs/training.log`.

### 2. Resume Training
If training was interrupted or you wish to extend it, you can resume training from the last saved state:
1. Ensure the checkpoint file `models/plate_detector/last.pt` exists.
2. Open `training/config.py` and set `RESUME: bool = True`.
3. Run the training script:
   ```powershell
   python training/train.py
   ```
   The script will detect the configuration, load `models/plate_detector/last.pt`, and pass `resume=True` to the Ultralytics training engine to continue seamlessly.

### 3. Model Evaluation
To evaluate your fine-tuned model against the validation dataset:
```powershell
python training/evaluate.py
```
- **What it does:**
  - Loads `models/plate_detector/best.pt`.
  - Prints system hardware info and model parameters.
  - Evaluates validation dataset split and logs precision, recall, mAP@50, and mAP@50-95.
  - Outputs speed metrics (preprocess, inference, postprocess, total latency).
  - Logs output directly to `outputs/logs/validation.log`.

### 4. Running Predictions
To visually inspect model performance on sample test images:
```powershell
python training/predict.py
```
- **What it does:**
  - Loads `best.pt` and runs predictions on up to 10 sample images from the test split.
  - Logs bounding box and class confidence statistics.
  - Saves annotated visualizations inside `outputs/predictions/`.
  - Logs output directly to `outputs/logs/prediction.log`.

### 5. Model Exporting
To export your trained weights (`best.pt`) to downstream production formats:
```powershell
python training/export.py
```
- **What it does:**
  - Exports to ONNX format: `models/plate_detector/best.onnx` (Dynamic shape, opset 12, simplified).
  - Exports to TensorRT: `models/plate_detector/best.engine` (FP16 optimized - *optional, requires local TensorRT installation*).
  - Logs output to `outputs/logs/export.log`.

---

## Expected Outputs & Folders

- **`models/plate_detector/`**:
  - `best.pt`: PyTorch weights achieving the highest validation mAP.
  - `last.pt`: Last saved epoch checkpoint (used for resuming training).
  - `best.onnx`: Exported ONNX model.
  - `best.engine`: Exported TensorRT engine (if available).
- **`outputs/`**:
  - `training_config.yaml`: YAML backup of hyperparameters, settings, and date of training.
  - `results.csv`: Metrics per epoch (loss, precision, recall, mAP).
  - `results.png`: Plotted training and validation loss curves.
  - `confusion_matrix.png` / `confusion_matrix_normalized.png`: Confusion matrices.
  - `BoxF1_curve.png` / `BoxPR_curve.png`: Precision and Recall curves.
  - `predictions/`: Visualization directory for test images.
  - **`logs/`**:
    - `training.log`: Run log file for `train.py`.
    - `validation.log`: Run log file for `evaluate.py`.
    - `prediction.log`: Run log file for `predict.py`.
    - `export.log`: Run log file for `export.py`.

---

## Troubleshooting & FAQ

1. **Multiprocessing Hangs on Windows:**
   - **Symptom:** Script starts but hangs indefinitely before epoch 1.
   - **Solution:** Ensure `workers=0` in `train.py` (our refactored script automatically does this on Windows).
2. **CUDA Out of Memory (OOM):**
   - **Symptom:** `RuntimeError: CUDA out of memory`.
   - **Solution:** Decrease `BATCH_SIZE` in `config.py` from `16` to `8` or `4`.
3. **Orphan Labels Found during Validation:**
   - **Symptom:** Dataset validation logs warnings about labels without matching images.
   - **Solution:** Run validation check to see which label files have no matching image, and clean them from `datasets/labels/` or place matching images in `datasets/images/`.

---

## Transition to Phase 2 (Inference Pipeline)

The fine-tuned weight file at `models/plate_detector/best.pt` serves as the primary detector artifact for Phase 2.
In **Phase 2 (Inference & OCR)**:
1. `src/plate_detector.py` will load `best.pt` (or `best.onnx` / `best.engine`) to isolate the license plate.
2. A vehicle detector will locate the vehicle boundaries.
3. The cropped vehicle region will be sent to the plate detector (`best.pt`) to locate and crop the license plate.
4. The cropped plate will be preprocessed (denoising, perspective correction, binarization) and fed into PP-OCRv4 to extract license plate text.

