"""Root execution runner for testing the ANPR Phase 3 Inference Pipeline.

Accepts an image path from the CLI, resolves the CUDA hardware device, runs end-to-end
predictions, saves annotated and JSON files, persists to MySQL if online, and outputs
the standardized ANPR JSON response to stdout.
"""

import sys
import json
import argparse
from pathlib import Path
import torch

# Add root folder to path to allow src package imports
sys.path.append(str(Path(__file__).resolve().parent))

from services.detection_service import DetectionService
from database.database import SessionLocal, is_db_connected

def main():
    parser = argparse.ArgumentParser(description="ANPR End-to-End Inference Pipeline Test Script")
    parser.add_argument(
        "image_path", 
        type=str, 
        help="Path to the single input image file (jpg, jpeg, png, bmp)"
    )
    parser.add_argument(
        "--device", 
        type=str, 
        default="auto", 
        help="Target processing hardware: 'cpu', '0' (for cuda:0), or 'auto' (default)"
    )

    args = parser.parse_args()

    # Resolve 'auto' device flag
    device_val = args.device
    if device_val == "auto":
        device_val = "0" if torch.cuda.is_available() else "cpu"

    image_path = Path(args.image_path)
    if not image_path.exists():
        print(f"Error: Input image file not found at: {image_path}", file=sys.stderr)
        sys.exit(1)

    try:
        # Read raw file bytes
        with open(image_path, "rb") as f:
            contents = f.read()

        # Initialize core detection service
        service = DetectionService()
        
        # Setup temporary DB session if database is online
        db = None
        if is_db_connected and SessionLocal is not None:
            db = SessionLocal()

        try:
            # Process image (will run pipeline, draw overlays, write JSON, write DB)
            result = service.process_image(contents, image_path.name, db)
            
            # Print final formatted JSON outputs to stdout
            print(json.dumps(result, indent=4))
        finally:
            if db:
                db.close()
                
    except Exception as e:
        print(f"Pipeline execution encountered an error: {str(e)}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

