"""Utility functions for upload validation, image checks, and user feedback."""

import io
from pathlib import Path

from PIL import Image
import streamlit as st

# ─────────────────────────────────────────────────────────────────────────────
# Image validation and feedback (French-first)
# ─────────────────────────────────────────────────────────────────────────────

VALID_FORMATS = {"jpg", "jpeg", "png"}
MIN_IMAGE_WIDTH = 100
MIN_IMAGE_HEIGHT = 100
MAX_FILE_SIZE_MB = 50


def validate_image_file(file_obj, name: str = None) -> tuple[bool, str | None]:
    """Validate uploaded image file for format, size, and integrity.
    
    Args:
        file_obj: Streamlit UploadedFile object
        name: Optional display name for error messages
        
    Returns:
        (is_valid, error_message): If valid, error_message is None.
                                   If invalid, is_valid is False and error_message is French feedback.
    """
    if not file_obj:
        return False, "Aucun fichier fourni."
    
    name = name or file_obj.name
    file_size_mb = file_obj.size / (1024 * 1024)
    
    # Check file extension
    ext = Path(file_obj.name).suffix.lower().lstrip(".")
    if ext not in VALID_FORMATS:
        return False, f"Format non supporté: {ext}. Utilisez JPG, JPEG ou PNG."
    
    # Check file size
    if file_size_mb > MAX_FILE_SIZE_MB:
        return False, f"Fichier trop volumineux: {file_size_mb:.1f} MB (max {MAX_FILE_SIZE_MB} MB)."
    
    # Try to load as image to check integrity
    try:
        file_bytes = file_obj.getvalue()
        img = Image.open(io.BytesIO(file_bytes))
        img.verify()
    except Exception as e:
        return False, f"Image corrompue ou non valide: {str(e)}"
    
    # Re-open after verify (verify() closes the file)
    try:
        img = Image.open(io.BytesIO(file_bytes))
        width, height = img.size
    except Exception:
        return False, "Impossible de lire les dimensions de l'image."
    
    # Check minimum dimensions
    if width < MIN_IMAGE_WIDTH or height < MIN_IMAGE_HEIGHT:
        return False, f"Image trop petite: {width}×{height}px (min {MIN_IMAGE_WIDTH}×{MIN_IMAGE_HEIGHT}px)."
    
    return True, None


def validate_images_batch(files_list: list) -> tuple[list, list]:
    """Validate a batch of uploaded images. Return valid and invalid file lists.
    
    Args:
        files_list: List of Streamlit UploadedFile objects
        
    Returns:
        (valid_files, invalid_files): 
            valid_files: List of valid file objects
            invalid_files: List of tuples (file_name, error_message)
    """
    valid = []
    invalid = []
    
    for f in files_list:
        is_valid, error_msg = validate_image_file(f)
        if is_valid:
            valid.append(f)
        else:
            invalid.append((f.name, error_msg))
    
    return valid, invalid


def show_validation_errors(invalid_files: list) -> None:
    """Display validation error messages for invalid files.
    
    Args:
        invalid_files: List of tuples (file_name, error_message)
    """
    if not invalid_files:
        return
    
    with st.expander(f"⚠️ {len(invalid_files)} fichier(s) rejeté(s)", expanded=False):
        for filename, error_msg in invalid_files:
            st.warning(f"**{filename}**: {error_msg}")


def show_validation_summary(valid_count: int, total_count: int) -> None:
    """Display a summary of validation results.
    
    Args:
        valid_count: Number of valid files
        total_count: Total files processed
    """
    if valid_count == total_count:
        st.success(f"✓ {total_count} image(s) valide(s).")
    elif valid_count > 0:
        st.warning(f"✓ {valid_count}/{total_count} image(s) valide(s). {total_count - valid_count} rejeté(e)s.")
    else:
        st.error(f"✗ Aucune image valide parmi {total_count} fichier(s).")
