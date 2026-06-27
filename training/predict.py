"""Prediction script for running inference with the fine-tuned plate detector.

Loads 'best.pt', selects sample images from the test dataset, runs predictions,
and saves the annotated output images.
"""

import os
import sys
import shutil
from pathlib import Path
import yaml
from ultralytics import YOLO

# Add the workspace root to sys.path to allow absolute imports
sys.path.append(str(Path(__file__).resolve().parent.parent))

from training import config
from training import utils

# Logger will be initialized dynamically in main()
logger = None


def main():
    """Performs inference on sample images and saves annotations."""
    global logger
    
    # Initialize logger
    logger = utils.setup_logger("PredictPipeline", "prediction.log", Path(config.LOGS_DIR))
    
    weights_path = config.MODELS_DIR / "best.pt"
    if not weights_path.exists():
        logger.error(f"Trained weights not found at: {weights_path}")
        logger.error("Please run training/train.py first.")
        sys.exit(1)

    local_yaml_path = Path(config.LOCAL_DATA_YAML)
    if not local_yaml_path.exists():
        logger.error(f"Local dataset configuration YAML not found at: {local_yaml_path}")
        logger.error("Please run training/train.py first.")
        sys.exit(1)

    # Load local yaml to identify test/validation image directories
    with open(local_yaml_path, "r", encoding="utf-8") as f:
        data_cfg = yaml.safe_load(f)

    # Try to find images in test directory, fallback to val and train
    datasets_root = Path(data_cfg["path"])
    
    test_dir_candidates = []
    if "test" in data_cfg:
        test_dir_candidates.append(datasets_root / data_cfg["test"])
    if "val" in data_cfg:
        test_dir_candidates.append(datasets_root / data_cfg["val"])
    if "train" in data_cfg:
        test_dir_candidates.append(datasets_root / data_cfg["train"])

    image_dir = None
    for candidate in test_dir_candidates:
        if candidate.exists() and any(candidate.iterdir()):
            image_dir = candidate
            break

    if not image_dir:
        logger.error("No image folders containing files found in the dataset.")
        sys.exit(1)

    logger.info(f"Using image source directory: {image_dir}")

    # Gather test images
    valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    image_files = [f for f in image_dir.iterdir() if f.suffix.lower() in valid_exts]
    
    if not image_files:
        logger.error(f"No valid image files found in directory: {image_dir}")
        sys.exit(1)

    # Limit to maximum 10 sample images for visualization
    sample_images = image_files[:10]
    logger.info(f"Found {len(image_files)} images, running prediction on {len(sample_images)} samples.")

    # Initialize model
    logger.info(f"Loading best weights: {weights_path}")
    try:
        model = YOLO(str(weights_path))
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        sys.exit(1)

    # Prepare outputs directories
    predictions_output_dir = Path(config.OUTPUT_DIRECTORY) / "predictions"
    os.makedirs(predictions_output_dir, exist_ok=True)

    # Resolve 'auto' device selection
    device_val = config.DEVICE
    if device_val == "auto":
        import torch
        device_val = "0" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device for prediction: {device_val}")

    # Predict and save outputs
    logger.info(f"Saving predictions to: {predictions_output_dir}")
    
    try:
        for idx, img_path in enumerate(sample_images):
            logger.info(f"Processing [{idx + 1}/{len(sample_images)}]: {img_path.name}")
            
            # Perform inference
            results = model.predict(
                source=str(img_path),
                imgsz=config.IMG_SIZE,
                device=device_val,
                conf=0.25,      # Confidence threshold
                save=True,      # Save annotated images automatically
                project=str(predictions_output_dir.parent),
                name="predictions_run",
                exist_ok=True
            )
            
            # Print bounding box details if any
            for r in results:
                boxes = r.boxes
                if len(boxes) > 0:
                    logger.info(f"Found {len(boxes)} license plate(s) in {img_path.name}:")
                    for box in boxes:
                        cls_idx = int(box.cls[0])
                        class_name = model.names[cls_idx]
                        conf_score = float(box.conf[0])
                        xyxy = box.xyxy[0].tolist()
                        logger.info(f"  - Class: {class_name}, Confidence: {conf_score:.2f}, Box: {[round(c, 2) for c in xyxy]}")
                else:
                    logger.info(f"No license plates detected in {img_path.name}.")

        # After inference is completed, locate saved predictions and structure them cleanly
        run_save_dir = Path(config.OUTPUT_DIRECTORY) / "predictions_run"
        if run_save_dir.exists():
            for item in run_save_dir.iterdir():
                if item.is_file():
                    # Move directly to outputs/predictions
                    shutil.copy2(item, predictions_output_dir / item.name)
            
            # Cleanup temporary run directory
            try:
                shutil.rmtree(run_save_dir)
            except Exception:
                pass

        logger.info(f"Prediction visualizations saved successfully to {predictions_output_dir}")

    except Exception as e:
        logger.error(f"Prediction process encountered an error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()

