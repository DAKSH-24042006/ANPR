"""Configuration file for ANPR YOLO11s model training and evaluation.

Defines all hyperparameters, paths, hardware configurations, and seeds.
"""

from pathlib import Path

# Base directories
BASE_DIR = Path(__file__).resolve().parent.parent

# Model settings
MODEL_NAME: str = "yolo11s.pt"

# Dataset settings
# Points to data.yaml in the datasets directory (which is linked to the Roboflow dataset)
DATASET_PATH: str = str(BASE_DIR / "datasets" / "data.yaml")

# Hyperparameters
IMG_SIZE: int = 640
EPOCHS: int = 100
BATCH_SIZE: int = 16
SEED: int = 42
EARLY_STOPPING_PATIENCE: int = 20

# Hardware settings
# "auto" will select CUDA automatically if available, falling back to CPU.
# Under Windows with RTX 4050, it will use CUDA.
DEVICE: str = "auto"

# Resume training settings
RESUME: bool = False

# Output settings
PROJECT_NAME: str = "ANPR_Plate_Detector"
OUTPUT_DIRECTORY: str = str(BASE_DIR / "outputs")
MODELS_DIR: Path = BASE_DIR / "models" / "plate_detector"
LOGS_DIR: Path = BASE_DIR / "outputs" / "logs"

# Temporary local dataset YAML file that resolves absolute paths
LOCAL_DATA_YAML: str = str(BASE_DIR / "outputs" / "data_local.yaml")
