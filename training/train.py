"""Training script for fine-tuning YOLO11s on the Indian License Plate Dataset.

Validates the dataset structure and file integrity, prepares a localized configuration,
performs transfer learning with official weights, and saves output models.
"""

import os
import sys
import shutil
import yaml
import torch
from pathlib import Path
from ultralytics import YOLO

# Add the workspace root to sys.path to allow absolute imports
sys.path.append(str(Path(__file__).resolve().parent.parent))

from training import config
from training import utils

# Logger will be initialized dynamically in main()
logger = None


def validate_and_prepare_dataset() -> str:
    """Performs validation on the dataset and creates a local YAML configuration file.

    Returns:
        str: Absolute path to the generated local YAML configuration.

    Raises:
        FileNotFoundError: If the source config or folders are missing.
        ValueError: If validation errors are found in the dataset.
    """
    logger.info(f"Checking dataset config at: {config.DATASET_PATH}")
    
    yaml_path = Path(config.DATASET_PATH)
    if not yaml_path.exists():
        raise FileNotFoundError(f"Dataset config YAML not found: {yaml_path}")

    # Load original dataset configuration
    with open(yaml_path, "r", encoding="utf-8") as f:
        data_cfg = yaml.safe_load(f)

    # Perform bidirectional dataset validation
    is_valid = utils.validate_dataset(yaml_path, logger)
    if not is_valid:
        raise ValueError("Dataset validation failed with errors. Please check the log messages above.")

    logger.info("Dataset validation passed successfully!")

    # Generate the local data_local.yaml with absolute paths to ensure robustness
    datasets_root = yaml_path.parent
    local_cfg = {
        "path": str(datasets_root.resolve()),
        "train": data_cfg["train"],
        "val": data_cfg["val"],
        "nc": data_cfg["nc"],
        "names": data_cfg["names"]
    }
    if "test" in data_cfg:
        local_cfg["test"] = data_cfg["test"]

    # Write local YAML configuration
    local_yaml_path = Path(config.LOCAL_DATA_YAML)
    os.makedirs(local_yaml_path.parent, exist_ok=True)
    with open(local_yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(local_cfg, f, default_flow_style=False)
        
    logger.info(f"Generated local dataset configuration at: {local_yaml_path}")
    return str(local_yaml_path.resolve())


def main():
    """Main training execution block."""
    global logger
    
    # Initialize timestamped logging inside outputs/logs/training.log
    logger = utils.setup_logger("TrainPipeline", "training.log", Path(config.LOGS_DIR))
    
    logger.info("Starting Automatic Number Plate Recognition (ANPR) Training Pipeline (Phase 1)")
    
    # Save training configuration backup inside outputs/
    utils.save_training_config(config, Path(config.OUTPUT_DIRECTORY))
    logger.info(f"Saved training configuration to: {Path(config.OUTPUT_DIRECTORY) / 'training_config.yaml'}")

    # Resolve training device
    device_val = config.DEVICE
    if device_val == "auto":
        device_val = "0" if torch.cuda.is_available() else "cpu"
        
    # Print CUDA/GPU detection details
    utils.print_gpu_info(device_val, logger)

    # Print dataset summary statistics before validation
    utils.print_dataset_statistics(Path(config.DATASET_PATH), logger)

    # Validate dataset and generate local yaml
    try:
        local_yaml_path = validate_and_prepare_dataset()
    except Exception as e:
        logger.error(f"Initialization/Dataset validation error: {str(e)}")
        sys.exit(1)

    # Check if models/plate_detector/last.pt exists for resume training support
    last_weights_path = config.MODELS_DIR / "last.pt"
    resume_training = False

    if config.RESUME:
        if last_weights_path.exists():
            resume_training = True
            model_path = last_weights_path
            logger.info("==================================================")
            logger.info(f"RESUMING TRAINING from last checkpoint: {model_path}")
            logger.info("==================================================")
        else:
            model_path = config.MODEL_NAME
            logger.warning("==================================================")
            logger.warning(f"Resume requested but last checkpoint not found at: {last_weights_path}")
            logger.warning(f"Starting FRESH training from pretrained weights: {model_path}")
            logger.warning("==================================================")
    else:
        model_path = config.MODEL_NAME
        logger.info("==================================================")
        logger.info(f"Starting FRESH training from pretrained weights: {model_path}")
        logger.info("==================================================")

    # Load model weights
    try:
        model = YOLO(str(model_path))
    except Exception as e:
        logger.error(f"Failed to load YOLO model: {str(e)}")
        sys.exit(1)

    # Prepare model destination directory
    os.makedirs(config.MODELS_DIR, exist_ok=True)

    # Start transfer learning / training
    logger.info("Initiating model training (transfer learning)...")
    try:
        results = model.train(
            data=local_yaml_path,
            epochs=config.EPOCHS,
            imgsz=config.IMG_SIZE,
            batch=config.BATCH_SIZE,
            device=device_val,
            seed=config.SEED,
            patience=config.EARLY_STOPPING_PATIENCE,
            project=config.PROJECT_NAME,
            name="train_run",
            val=True,
            amp=True,
            exist_ok=True,
            save=True,
            resume=resume_training,
            workers=0 if os.name == 'nt' else 8  # Use 0 workers on Windows to prevent multi-processing overhead/hangs
        )
        logger.info("Training process completed successfully.")
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        sys.exit(1)

    # Locate and copy final weights using official Ultralytics API save directory
    try:
        run_dir = Path(results.save_dir)
        logger.info(f"Resolving training run outputs from save directory: {run_dir}")
        
        best_weights = run_dir / "weights" / "best.pt"
        last_weights = run_dir / "weights" / "last.pt"

        if best_weights.exists() and last_weights.exists():
            # Copy weights to target models/plate_detector directory
            shutil.copy2(best_weights, config.MODELS_DIR / "best.pt")
            shutil.copy2(last_weights, config.MODELS_DIR / "last.pt")
            logger.info(f"Successfully copied final weights to {config.MODELS_DIR}")
            
            # Copy training results/plots to outputs/ for quick access
            copied_count = 0
            for item in run_dir.iterdir():
                if item.is_file() and item.suffix in [".csv", ".png", ".jpg", ".txt"]:
                    shutil.copy2(item, config.OUTPUT_DIRECTORY)
                    copied_count += 1
            logger.info(f"Successfully saved {copied_count} validation reports/curves to {config.OUTPUT_DIRECTORY}")
        else:
            logger.error(f"Expected weights not found under {run_dir / 'weights'}. Training output copy failed.")
            sys.exit(1)
    except Exception as e:
        logger.error(f"Post-training weight copy encountered error: {str(e)}")
        sys.exit(1)

    logger.info("Phase 1 - Training Pipeline Completed.")


if __name__ == "__main__":
    main()
