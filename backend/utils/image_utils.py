import os
import uuid
import shutil
from pathlib import Path
from fastapi import UploadFile

MAX_FILE_SIZE_MB = 20
ALLOWED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.tiff', '.bmp'}

def generate_unique_filename(original_name: str) -> str:
    """Generate a unique filename using UUID, preserving extension."""
    ext = os.path.splitext(original_name)[1].lower()
    return f"{uuid.uuid4()}{ext}"

def validate_image_file(file: UploadFile) -> bool:
    """Validate the uploaded image file extension and size (size handled implicitly if needed)."""
    ext = os.path.splitext(file.filename)[1].lower()
    return ext in ALLOWED_EXTENSIONS

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
UPLOAD_DIR = os.path.abspath(PROJECT_ROOT / "uploads")

def save_upload_file(upload_file: UploadFile, upload_dir: str = None) -> str:
    """Save an uploaded file to the specified directory and return the path."""
    target_dir = upload_dir or UPLOAD_DIR
    os.makedirs(target_dir, exist_ok=True)
    filename = generate_unique_filename(upload_file.filename)
    file_path = os.path.join(target_dir, filename)
    
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(upload_file.file, buffer)
        
    return file_path

def cleanup_temp_files(paths: list[str]):
    """Clean up temporary files."""
    for path in paths:
        if path and os.path.exists(path):
            try:
                os.remove(path)
            except Exception as e:
                print(f"Failed to remove temp file {path}: {e}")
