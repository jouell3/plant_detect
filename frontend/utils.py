"""Utility functions for upload validation, image checks, and user feedback."""

import io
from pathlib import Path
import threading
from typing import Callable

from PIL import Image
import requests
import streamlit as st
from loguru import logger

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


def get_streamlit_session_id() -> str:
    """Return current Streamlit session id, or a safe fallback."""
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx

        ctx = get_script_run_ctx()
        return ctx.session_id if ctx else "default"
    except Exception:
        return "default"


@st.cache_resource
def get_batch_bg_state(namespace: str) -> dict:
    """Return a cached shared background state bucket for a namespace."""
    return {
        "namespace": namespace,
        "lock": threading.Lock(),
        "results": {},   # session_id -> {filename: result}
        "running": set(),
        "progress": {},  # session_id -> {done:int, total:int, errors:int}
        "failed_files": {},  # session_id -> list[dict] (files from failed chunks)
    }


def chunk_files(files: list[dict], chunk_size: int) -> list[list[dict]]:
    """Split files into fixed-size chunks."""
    return [files[i : i + chunk_size] for i in range(0, len(files), chunk_size)]


def clear_batch_session_tracking(bg_state: dict, session_id: str) -> None:
    """Clear background tracking for a given user session."""
    with bg_state["lock"]:
        bg_state["running"].discard(session_id)
        bg_state["results"].pop(session_id, None)
        bg_state["progress"].pop(session_id, None)
        bg_state["failed_files"].pop(session_id, None)


def reset_batch_page_state(
    session_id: str,
    bg_state: dict,
    image_files_key: str,
    batch_results_key: str,
    batches_loaded_key: str,
    page_key: str,
    cache_clear_fn: Callable[[], None],
) -> None:
    """Reset Streamlit state keys and background tracking for a batch page."""
    st.session_state[image_files_key] = []
    st.session_state[batch_results_key] = {}
    st.session_state[batches_loaded_key] = set()
    st.session_state[page_key] = 0
    cache_clear_fn()
    clear_batch_session_tracking(bg_state, session_id)


def run_sequential_subbatch_fetch(
    session_id: str,
    files: list[dict],
    chunk_size: int,
    fetch_batch_fn: Callable[[list[dict]], dict],
    bg_state: dict,
    log_prefix: str,
) -> None:
    """Background worker that fetches chunks sequentially and tracks progress."""
    chunks = chunk_files(files, chunk_size)
    total = len(chunks)
    with bg_state["lock"]:
        bg_state["progress"][session_id] = {"done": 0, "total": total, "errors": 0}
        bg_state["failed_files"][session_id] = []

    if not chunks:
        with bg_state["lock"]:
            bg_state["running"].discard(session_id)
        return

    try:
        for chunk in chunks:
            try:
                chunk_results = fetch_batch_fn(chunk)
                with bg_state["lock"]:
                    bg_state["results"].setdefault(session_id, {}).update(chunk_results)
            except Exception as e:
                logger.warning("{} | {}", log_prefix, e)
                with bg_state["lock"]:
                    bg_state["progress"][session_id]["errors"] += 1
                    bg_state["failed_files"][session_id].extend(chunk)
            finally:
                with bg_state["lock"]:
                    bg_state["progress"][session_id]["done"] += 1
    finally:
        with bg_state["lock"]:
            bg_state["running"].discard(session_id)


def render_batch_lot_grids(
    *,
    all_files: list[dict],
    batch_results: dict,
    page_size: int,
    grid_cols: int,
    render_item_fn: Callable[[dict, dict], None],
    pending_caption: str = "Chargement des images restantes de ce lot…",
) -> None:
    """Render loaded batch predictions as chunked lots with a shared layout."""
    total_files = len(all_files)
    for chunk_start in range(0, total_files, page_size):
        chunk_end = min(chunk_start + page_size, total_files)
        chunk = all_files[chunk_start:chunk_end]
        chunk_loaded = [f for f in chunk if f["name"] in batch_results]
        lot_num = (chunk_start // page_size) + 1

        st.markdown(f"### Lot {lot_num} ({len(chunk_loaded)}/{len(chunk)})")

        for row_idx in range(0, len(chunk_loaded), grid_cols):
            cols = st.columns(grid_cols)
            for col_idx, file in enumerate(chunk_loaded[row_idx : row_idx + grid_cols]):
                with cols[col_idx]:
                    render_item_fn(file, batch_results[file["name"]])

        if len(chunk_loaded) < len(chunk):
            st.caption(pending_caption)
        st.divider()


def render_batch_progress_footer(*, loaded_total: int, total_files: int, is_running: bool, progress: dict) -> None:
    """Render standard progress footer for sequential sub-batch loading."""
    if is_running:
        st.caption(
            f"{loaded_total} / {total_files} images affichées — "
            f"requêtes séquentielles en cours ({progress['done']}/{progress['total']} lots terminés, erreurs: {progress['errors']})."
        )
    elif loaded_total < total_files:
        st.warning(
            f"Chargement interrompu: {loaded_total}/{total_files} images reçues. "
            "Certaines requêtes ont échoué; relancez uniquement les lots échoués."
        )
    else:
        st.caption(f"Toutes les {total_files} images ont été chargées.")


def post_with_retries(
    *,
    url: str,
    files,
    timeout: int,
    retry_delays_seconds: tuple[float, ...],
    log_message: str,
):
    """POST helper with retry/backoff for connection/timeout/http errors."""
    last_error = None
    for idx, delay in enumerate((0.0, *retry_delays_seconds)):
        if delay > 0:
            import time

            time.sleep(delay)
        try:
            response = requests.post(url, files=files, timeout=timeout)
            response.raise_for_status()
            return response
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout, requests.exceptions.HTTPError) as e:
            last_error = e
            logger.warning("{} | attempt={} | error={}", log_message, idx + 1, e)
    raise last_error
