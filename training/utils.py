"""Utility module for ANPR YOLO11s model training and validation.

Houses reusable helpers for GPU identification, bidirectional dataset validation,
dataset statistics reporting, logging configuration, and configuration backups.
"""

import os
import sys
import logging
from pathlib import Path
from datetime import datetime
import yaml
from PIL import Image

def print_gpu_info(device_str: str, logger=None):
    """Detects and prints detailed system CUDA and GPU information.

    Args:
        device_str: Resolved training device string.
        logger: Optional logger to output info.
    """
    import torch
    cuda_available = torch.cuda.is_available()
    pytorch_version = torch.__version__
    python_version = sys.version.split()[0]
    
    info_lines = [
        "=================================",
        f"CUDA      : {'Available' if cuda_available else 'Not Available'}"
    ]
    
    if cuda_available:
        gpu_name = torch.cuda.get_device_name(0)
        total_memory = torch.cuda.get_device_properties(0).total_memory
        vram_gb = round(total_memory / (1024 ** 3))
        cuda_version = torch.version.cuda
        info_lines.extend([
            f"GPU       : {gpu_name}",
            f"VRAM      : {vram_gb} GB",
            f"CUDA Ver  : {cuda_version}"
        ])
    else:
        info_lines.extend([
            "GPU       : None",
            "VRAM      : None"
        ])
        
    info_lines.extend([
        f"PyTorch   : {pytorch_version}",
        f"Python    : {python_version}",
        f"Device    : {device_str}",
        "================================="
    ])
    
    output_str = "\n" + "\n".join(info_lines) + "\n"
    if logger:
        logger.info(output_str)
    else:
        print(output_str)


def print_dataset_statistics(yaml_path: Path, logger=None):
    """Parses data.yaml, computes dataset split counts, and resolution statistics.

    Args:
        yaml_path: Path to dataset data.yaml.
        logger: Optional logger to output statistics.
    """
    if not yaml_path.exists():
        msg = f"Dataset configuration file not found at: {yaml_path}"
        if logger:
            logger.error(msg)
        else:
            print(msg)
        return

    with open(yaml_path, "r", encoding="utf-8") as f:
        data_cfg = yaml.safe_load(f)

    datasets_root = yaml_path.parent
    splits = ["train", "val", "test"]
    valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    
    info_lines = [
        "=================================",
        "DATASET STATISTICS SUMMARY",
        "================================="
    ]
    
    total_images_all = 0
    
    for split in splits:
        split_key = "val" if split == "val" else split
        # Fallback for naming variations
        if split_key not in data_cfg and split == "val":
            split_key = "validation"
        if split_key not in data_cfg:
            continue
            
        img_relative = data_cfg[split_key]
        img_dir = datasets_root / img_relative
        lbl_dir = datasets_root / img_relative.replace("images", "labels")
        
        img_count = 0
        lbl_count = 0
        
        if img_dir.exists():
            img_count = len([f for f in img_dir.iterdir() if f.suffix.lower() in valid_exts])
        if lbl_dir.exists():
            lbl_count = len([f for f in lbl_dir.iterdir() if f.suffix.lower() == ".txt"])
            
        total_images_all += img_count
        info_lines.append(f"{split.capitalize()} Split:")
        info_lines.append(f"  Images : {img_count}")
        info_lines.append(f"  Labels : {lbl_count}")
        
        # Calculate resolution range
        widths = []
        heights = []
        if img_dir.exists():
            for file in img_dir.iterdir():
                if file.suffix.lower() in valid_exts:
                    try:
                        with Image.open(file) as img:
                            w, h = img.size
                            widths.append(w)
                            heights.append(h)
                    except Exception:
                        pass
        if widths:
            min_w, max_w, avg_w = min(widths), max(widths), sum(widths) // len(widths)
            min_h, max_h, avg_h = min(heights), max(heights), sum(heights) // len(heights)
            info_lines.append(f"  Resolutions : {min_w}x{min_h} to {max_w}x{max_h} (Avg: {avg_w}x{avg_h})")
            
    info_lines.extend([
        f"Total Images     : {total_images_all}",
        f"Number of Classes: {data_cfg.get('nc', 0)}",
        f"Class Names      : {data_cfg.get('names', [])}",
        "================================="
    ])
    
    output_str = "\n" + "\n".join(info_lines) + "\n"
    if logger:
        logger.info(output_str)
    else:
        print(output_str)


def validate_dataset(yaml_path: Path, logger) -> bool:
    """Performs strict bidirectional verification (Image -> Label & Label -> Image).

    Finds missing, corrupt, malformed files, and orphan labels.

    Args:
        yaml_path: Path to dataset data.yaml.
        logger: Logger for writing errors and summaries.

    Returns:
        bool: True if dataset has no corrupt/orphan/missing elements.
    """
    logger.info("=" * 40)
    logger.info("DATASET BIDIRECTIONAL VALIDATION")
    logger.info("=" * 40)

    if not yaml_path.exists():
        logger.error(f"Dataset config YAML not found: {yaml_path}")
        return False

    with open(yaml_path, "r", encoding="utf-8") as f:
        data_cfg = yaml.safe_load(f)

    datasets_root = yaml_path.parent
    splits = ["train", "val", "test"]
    valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    
    overall_valid = True
    
    for split in splits:
        split_key = "val" if split == "val" else split
        if split_key not in data_cfg and split == "val":
            split_key = "validation"
        if split_key not in data_cfg:
            continue
            
        img_relative = data_cfg[split_key]
        img_dir = datasets_root / img_relative
        lbl_dir = datasets_root / img_relative.replace("images", "labels")
        
        logger.info(f"Validating split: {split}")
        
        if not img_dir.exists():
            logger.error(f"[{split}] Images folder missing: {img_dir}")
            overall_valid = False
            continue
        if not lbl_dir.exists():
            logger.error(f"[{split}] Labels folder missing: {lbl_dir}")
            overall_valid = False
            continue
            
        image_files = {f.stem: f for f in img_dir.iterdir() if f.suffix.lower() in valid_exts}
        label_files = {f.stem: f for f in lbl_dir.iterdir() if f.suffix.lower() == ".txt"}
        
        missing_images = 0
        missing_labels = 0
        corrupt_images = 0
        invalid_annotations = 0
        orphan_labels = []
        
        # 1. Image -> Label verification (and image corruption checks)
        for stem, img_path in image_files.items():
            # Check corrupt image
            try:
                with Image.open(img_path) as img:
                    img.verify()
            except Exception as e:
                logger.error(f"[{split}] Corrupt image file: {img_path.name}. Error: {str(e)}")
                corrupt_images += 1
                overall_valid = False
                continue
                
            # Check corresponding label
            label_filename = f"{stem}.txt"
            label_path = lbl_dir / label_filename
            
            if not label_path.exists():
                # YOLO supports background images (no labels), warn but don't fail validation
                missing_labels += 1
                logger.warning(f"[{split}] Image {img_path.name} has no matching label file (background image).")
                continue
                
            # Check annotation bounds and formats
            try:
                with open(label_path, "r", encoding="utf-8") as lf:
                    lines = lf.readlines()
                for idx, line in enumerate(lines):
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split()
                    
                    if len(parts) < 5 or len(parts) % 2 == 0:
                        logger.error(f"[{split}] Malformed label in {label_filename} (line {idx+1}): Expected >=5 values and odd length, got {len(parts)}")
                        invalid_annotations += 1
                        overall_valid = False
                        break
                        
                    try:
                        cls_id = int(parts[0])
                        coords = [float(x) for x in parts[1:]]
                        for coord in coords:
                            if not (0.0 <= coord <= 1.0):
                                logger.error(f"[{split}] Out-of-bounds coordinates in {label_filename} (line {idx+1}): Coordinate {coord} must be normalized between 0.0 and 1.0")
                                invalid_annotations += 1
                                overall_valid = False
                                break
                        if invalid_annotations > 0:
                            break
                    except ValueError:
                        logger.error(f"[{split}] Non-numeric values in {label_filename} (line {idx+1})")
                        invalid_annotations += 1
                        overall_valid = False
                        break
            except Exception as e:
                logger.error(f"[{split}] Failed to read label file {label_filename}. Error: {str(e)}")
                invalid_annotations += 1
                overall_valid = False
                
        # 2. Label -> Image verification (to find orphan labels)
        for stem, lbl_path in label_files.items():
            if stem not in image_files:
                logger.error(f"[{split}] Orphan label file found: {lbl_path.name} has no matching image.")
                orphan_labels.append(lbl_path.name)
                overall_valid = False
                
        logger.info(f"Split [{split}] Verification Report:")
        logger.info(f"  - Verified Images: {len(image_files)}")
        logger.info(f"  - Verified Labels: {len(label_files)}")
        logger.info(f"  - Corrupt Images Found: {corrupt_images}")
        logger.info(f"  - Invalid Annotations Found: {invalid_annotations}")
        logger.info(f"  - Orphan Labels (Labels without Images): {len(orphan_labels)}")
        logger.info(f"  - Background Images (Images without Labels): {missing_labels}")
        
    logger.info("=" * 40)
    return overall_valid


def setup_logger(log_name: str, log_filename: str, logs_dir: Path) -> logging.Logger:
    """Configures console and file logger inside outputs/logs/.

    Args:
        log_name: Logger namespace.
        log_filename: Target file name (e.g., training.log).
        logs_dir: Folder to save log files.

    Returns:
        logging.Logger instance.
    """
    os.makedirs(logs_dir, exist_ok=True)
    logger = logging.getLogger(log_name)
    logger.setLevel(logging.INFO)
    
    # Check if handlers are already configured to avoid duplication
    if not logger.handlers:
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        
        # Console handler
        ch = logging.StreamHandler(sys.stdout)
        ch.setFormatter(formatter)
        logger.addHandler(ch)
        
        # File handler (Overwrites log on fresh run)
        log_filepath = logs_dir / log_filename
        fh = logging.FileHandler(log_filepath, mode="w", encoding="utf-8")
        fh.setFormatter(formatter)
        logger.addHandler(fh)
        
    return logger


def save_training_config(config_module, output_dir: Path):
    """Backups current hyperparameters and settings into outputs/training_config.yaml.

    Args:
        config_module: Imported config.py module.
        output_dir: Outputs base folder to save YAML config.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    config_data = {
        "Model Name": config_module.MODEL_NAME,
        "Dataset Path": config_module.DATASET_PATH,
        "Epochs": config_module.EPOCHS,
        "Batch Size": config_module.BATCH_SIZE,
        "Image Size": config_module.IMG_SIZE,
        "Learning Rate": {
            "lr0": 0.01,
            "lrf": 0.01
        },
        "Optimizer": "auto",
        "Seed": config_module.SEED,
        "Device": config_module.DEVICE,
        "Date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    config_yaml_path = output_dir / "training_config.yaml"
    with open(config_yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(config_data, f, default_flow_style=False)
