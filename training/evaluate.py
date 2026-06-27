"""Evaluation script for verifying the trained license plate detector.

Loads 'best.pt', validates it on the dataset, and prints metrics (Precision,
Recall, mAP, Inference Speed, Model Size, and Parameters).
"""

import os
import sys
from pathlib import Path
from ultralytics import YOLO

# Add the workspace root to sys.path to allow absolute imports
sys.path.append(str(Path(__file__).resolve().parent.parent))

from training import config
from training import utils

# Logger will be initialized dynamically in main()
logger = None


def main():
    """Performs model evaluation and prints metrics."""
    global logger
    
    # Initialize logger
    logger = utils.setup_logger("ValidationPipeline", "validation.log", Path(config.LOGS_DIR))
    
    # Ensure weights exist
    weights_path = config.MODELS_DIR / "best.pt"
    if not weights_path.exists():
        logger.error(f"Trained model weights not found at: {weights_path}")
        logger.error("Please run training/train.py first.")
        sys.exit(1)

    # Resolve local dataset yaml
    local_yaml_path = Path(config.LOCAL_DATA_YAML)
    if not local_yaml_path.exists():
        logger.error(f"Local dataset YAML config not found at: {local_yaml_path}")
        logger.error("Please run training/train.py first to initialize validation config.")
        sys.exit(1)

    # Resolve 'auto' device selection
    device_val = config.DEVICE
    if device_val == "auto":
        import torch
        device_val = "0" if torch.cuda.is_available() else "cpu"

    logger.info("================================================")
    logger.info("ANPR Model Evaluation Pipeline (Phase 1)")
    logger.info("================================================")
    
    # Display GPU information
    utils.print_gpu_info(device_val, logger)
    
    logger.info(f"Loading best model weights from: {weights_path}")
    
    # Load model
    try:
        model = YOLO(str(weights_path))
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        sys.exit(1)

    # 1. Measure model size and parameters
    model_size_bytes = os.path.getsize(weights_path)
    model_size_mb = model_size_bytes / (1024 * 1024)

    # Get parameters count
    num_params = sum(p.numel() for p in model.model.parameters())

    logger.info(f"Number of Parameters: {num_params:,}")
    logger.info(f"Model File Size: {model_size_mb:.2f} MB")
    logger.info("-" * 50)

    # 2. Run validation/evaluation
    logger.info("Running validation split evaluation...")
    logger.info(f"Using device for evaluation: {device_val}")

    try:
        # Run validation
        results = model.val(
            data=str(local_yaml_path),
            device=device_val,
            plots=True,
            save_json=True
        )
    except Exception as e:
        logger.error(f"Validation failed: {str(e)}")
        sys.exit(1)

    # 3. Extract metrics
    metrics_dict = results.results_dict
    precision = metrics_dict.get("metrics/precision(B)", 0.0)
    recall = metrics_dict.get("metrics/recall(B)", 0.0)
    map50 = metrics_dict.get("metrics/mAP50(B)", 0.0)
    map50_95 = metrics_dict.get("metrics/mAP50-95(B)", 0.0)

    # Speeds are returned in ms per image
    speed_preprocess = results.speed.get("preprocess", 0.0)
    speed_inference = results.speed.get("inference", 0.0)
    speed_postprocess = results.speed.get("postprocess", 0.0)
    total_speed = speed_preprocess + speed_inference + speed_postprocess

    # Print summary metrics to console/logger
    logger.info("\n" + "=" * 50)
    logger.info("MODEL METRICS SUMMARY")
    logger.info("=" * 50)
    logger.info(f"Precision:         {precision:.4f} ({(precision * 100):.2f}%)")
    logger.info(f"Recall:            {recall:.4f} ({(recall * 100):.2f}%)")
    logger.info(f"mAP@50:            {map50:.4f} ({(map50 * 100):.2f}%)")
    logger.info(f"mAP@50-95:         {map50_95:.4f} ({(map50_95 * 100):.2f}%)")
    logger.info("-" * 50)
    logger.info("INFERENCE SPEED (per image)")
    logger.info("-" * 50)
    logger.info(f"Preprocess:        {speed_preprocess:.2f} ms")
    logger.info(f"Inference:         {speed_inference:.2f} ms")
    logger.info(f"Postprocess:       {speed_postprocess:.2f} ms")
    logger.info(f"Total Speed:       {total_speed:.2f} ms")
    logger.info("-" * 50)
    logger.info(f"Validation plots saved in: {results.save_dir}")
    logger.info("=" * 50 + "\n")


if __name__ == "__main__":
    main()
