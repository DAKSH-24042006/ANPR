"""API Configuration settings for ANPR service.

Stores database credentials, server ports, upload sizes, and formats.
Loads properties from environment variables for environment configuration.
"""

import os

# --- FastAPI server configurations ---
FASTAPI_HOST: str = os.getenv("FASTAPI_HOST", "127.0.0.1")
FASTAPI_PORT: int = int(os.getenv("FASTAPI_PORT", "8000"))

# --- MySQL database credentials ---
DB_HOST: str = os.getenv("DB_HOST", "localhost")
DB_PORT: int = int(os.getenv("DB_PORT", "3306"))
DB_USER: str = os.getenv("DB_USER", "root")
DB_PASSWORD: str = os.getenv("DB_PASSWORD", "root_password")
DB_NAME: str = os.getenv("DB_NAME", "anpr_db")

# --- Storage constraints ---
UPLOAD_FOLDER: str = os.getenv("UPLOAD_FOLDER", "outputs/original_images")
OUTPUT_FOLDER: str = os.getenv("OUTPUT_FOLDER", "outputs")
MAX_UPLOAD_SIZE: int = 10 * 1024 * 1024  # 10 MB in bytes

# Supported file extension list (lowercase with dots)
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}
