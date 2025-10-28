"""Validation utilities."""

from __future__ import annotations
from flask import Request
from typing import Tuple

MAX_UPLOAD_SIZE = 50 * 1024 * 1024  # 50MB


def validate_upload_request(request: Request) -> Tuple[str, str]:
    """Validate upload request and return (dataset_id, error_message)."""
    if 'file' not in request.files:
        return None, "No file provided"
    
    file = request.files['file']
    
    if file.filename == '':
        return None, "Empty filename"
    
    if not file.filename.endswith('.zip'):
        return None, "File must be a .zip archive"
    
    # Check file size
    file.seek(0, 2)  # SEEK_END
    file_size = file.tell()
    file.seek(0)
    
    if file_size > MAX_UPLOAD_SIZE:
        return None, f"File too large. Maximum size: {MAX_UPLOAD_SIZE / (1024*1024)}MB"
    
    dataset_id = request.form.get('dataset_id', 'default-dataset')
    return dataset_id, None
