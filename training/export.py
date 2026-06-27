"""Export script for converting the trained YOLO11s PyTorch model.

Loads 'best.pt' and exports it to production formats: ONNX and TensorRT (engine).
"""

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
    """Loads weights and exports to ONNX and TensorRT."""
    global logger
    
    # Initialize logger
    logger = utils.setup_logger("ExportPipeline", "export.log", Path(config.LOGS_DIR))
    
    weights_path = config.MODELS_DIR / "best.pt"
    if not weights_path.exists():
        logger.error(f"Trained weights not found at: {weights_path}")
        logger.error("Please run training/train.py first.")
        sys.exit(1)

    logger.info(f"Loading model from: {weights_path}")
    try:
        model = YOLO(str(weights_path))
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        sys.exit(1)

    # 1. Export to ONNX
    logger.info("Exporting model to ONNX format...")
    try:
        onnx_file = model.export(
            format="onnx",
            dynamic=True,       # Enable dynamic axes for sizing flexibility
            opset=12,          # Use stable opset 12
            simplify=True      # Run ONNX simplifier
        )
        logger.info(f"ONNX export completed. Output file: {onnx_file}")
    except Exception as e:
        logger.error(f"Failed to export to ONNX: {str(e)}")

    # 2. Export to TensorRT (Optional / if environment supports it)
    logger.info("Exporting model to TensorRT (engine) format...")
    try:
        # Resolve 'auto' device selection
        device_val = config.DEVICE
        if device_val == "auto":
            import torch
            device_val = "0" if torch.cuda.is_available() else "cpu"
        
        # Note: TensorRT export requires TensorRT SDK installed on the machine.
        # It will build a serialized engine optimized for the local RTX 4050 GPU.
        engine_file = model.export(
            format="engine",
            device=device_val,
            half=True          # Enable FP16 precision for RTX 4050 optimization
        )
        logger.info(f"TensorRT export completed. Output file: {engine_file}")
    except Exception as e:
        logger.warning(
            "TensorRT export skipped or failed. This is expected if the CUDA/TensorRT "
            f"environment is not fully configured on this machine. Details: {str(e)}"
        )

    logger.info(f"Export processing finished. Files are located in {config.MODELS_DIR}")


if __name__ == "__main__":
    main()

