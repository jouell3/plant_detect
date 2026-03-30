"""Utility functions for upload validation, image checks, and user feedback."""

import io
from pathlib import Path
import threading
from typing import Callable

from PIL import Image
import requests
import streamlit as st
from loguru import logger

from i18n import get_language

# ─────────────────────────────────────────────────────────────────────────────
# Image validation and feedback (French-first)
# ─────────────────────────────────────────────────────────────────────────────

VALID_FORMATS = {"jpg", "jpeg", "png"}
MIN_IMAGE_WIDTH = 100
MIN_IMAGE_HEIGHT = 100
MAX_FILE_SIZE_MB = 50


_MESSAGES = {
    "no_file": {
        "fr": "Aucun fichier fourni.",
        "en": "No file was provided.",
    },
    "unsupported_format": {
        "fr": "Format non supporte: {ext}. Utilisez JPG, JPEG ou PNG.",
        "en": "Unsupported format: {ext}. Use JPG, JPEG, or PNG.",
    },
    "file_too_large": {
        "fr": "Fichier trop volumineux: {size:.1f} MB (max {max_size} MB).",
        "en": "File is too large: {size:.1f} MB (max {max_size} MB).",
    },
    "invalid_image": {
        "fr": "Image corrompue ou non valide: {error}",
        "en": "Image is corrupted or invalid: {error}",
    },
    "cannot_read_dimensions": {
        "fr": "Impossible de lire les dimensions de l'image.",
        "en": "Unable to read image dimensions.",
    },
    "image_too_small": {
        "fr": "Image trop petite: {width}x{height}px (min {min_w}x{min_h}px).",
        "en": "Image is too small: {width}x{height}px (min {min_w}x{min_h}px).",
    },
    "rejected_files": {
        "fr": "⚠️ {count} fichier(s) rejete(s)",
        "en": "⚠️ {count} rejected file(s)",
    },
    "all_valid": {
        "fr": "✓ {total} image(s) valide(s).",
        "en": "✓ {total} valid image(s).",
    },
    "partially_valid": {
        "fr": "✓ {valid}/{total} image(s) valide(s). {rejected} rejetee(s).",
        "en": "✓ {valid}/{total} valid image(s). {rejected} rejected.",
    },
    "none_valid": {
        "fr": "✗ Aucune image valide parmi {total} fichier(s).",
        "en": "✗ No valid image found among {total} file(s).",
    },
    "pending_lot": {
        "fr": "Chargement des images restantes de ce lot...",
        "en": "Loading the remaining images for this batch...",
    },
    "lot_title": {
        "fr": "### Lot {num} ({loaded}/{total})",
        "en": "### Batch {num} ({loaded}/{total})",
    },
    "progress_running": {
        "fr": "{loaded} / {total} images affichees - requetes sequentielles en cours ({done}/{all_batches} lots termines, erreurs: {errors}).",
        "en": "{loaded} / {total} images displayed - sequential requests in progress ({done}/{all_batches} batches completed, errors: {errors}).",
    },
    "progress_interrupted": {
        "fr": "Chargement interrompu: {loaded}/{total} images recues. Certaines requetes ont echoue; relancez uniquement les lots echoues.",
        "en": "Loading interrupted: {loaded}/{total} images received. Some requests failed; retry only failed batches.",
    },
    "progress_done": {
        "fr": "Toutes les {total} images ont ete chargees.",
        "en": "All {total} images have been loaded.",
    },
}


def _msg(key: str, **kwargs) -> str:
    lang = get_language()
    template = _MESSAGES.get(key, {}).get(lang) or _MESSAGES.get(key, {}).get("fr") or key
    return template.format(**kwargs)


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
        return False, _msg("no_file")
    
    name = name or file_obj.name
    file_size_mb = file_obj.size / (1024 * 1024)
    
    # Check file extension
    ext = Path(file_obj.name).suffix.lower().lstrip(".")
    if ext not in VALID_FORMATS:
        return False, _msg("unsupported_format", ext=ext)
    
    # Check file size
    if file_size_mb > MAX_FILE_SIZE_MB:
        return False, _msg("file_too_large", size=file_size_mb, max_size=MAX_FILE_SIZE_MB)
    
    # Try to load as image to check integrity
    try:
        file_bytes = file_obj.getvalue()
        img = Image.open(io.BytesIO(file_bytes))
        img.verify()
    except Exception as e:
        return False, _msg("invalid_image", error=str(e))
    
    # Re-open after verify (verify() closes the file)
    try:
        img = Image.open(io.BytesIO(file_bytes))
        width, height = img.size
    except Exception:
        return False, _msg("cannot_read_dimensions")
    
    # Check minimum dimensions
    if width < MIN_IMAGE_WIDTH or height < MIN_IMAGE_HEIGHT:
        return False, _msg("image_too_small", width=width, height=height, min_w=MIN_IMAGE_WIDTH, min_h=MIN_IMAGE_HEIGHT)
    
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
    
    with st.expander(_msg("rejected_files", count=len(invalid_files)), expanded=False):
        for filename, error_msg in invalid_files:
            st.warning(f"**{filename}**: {error_msg}")


def show_validation_summary(valid_count: int, total_count: int) -> None:
    """Display a summary of validation results.
    
    Args:
        valid_count: Number of valid files
        total_count: Total files processed
    """
    if valid_count == total_count:
        st.success(_msg("all_valid", total=total_count))
    elif valid_count > 0:
        st.warning(_msg("partially_valid", valid=valid_count, total=total_count, rejected=total_count - valid_count))
    else:
        st.error(_msg("none_valid", total=total_count))


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
    pending_caption: str | None = None,
) -> None:
    """Render loaded batch predictions as chunked lots with a shared layout."""
    total_files = len(all_files)
    local_pending = pending_caption or _msg("pending_lot")
    for chunk_start in range(0, total_files, page_size):
        chunk_end = min(chunk_start + page_size, total_files)
        chunk = all_files[chunk_start:chunk_end]
        chunk_loaded = [f for f in chunk if f["name"] in batch_results]
        lot_num = (chunk_start // page_size) + 1

        st.markdown(_msg("lot_title", num=lot_num, loaded=len(chunk_loaded), total=len(chunk)))

        for row_idx in range(0, len(chunk_loaded), grid_cols):
            cols = st.columns(grid_cols)
            for col_idx, file in enumerate(chunk_loaded[row_idx : row_idx + grid_cols]):
                with cols[col_idx]:
                    render_item_fn(file, batch_results[file["name"]])

        if len(chunk_loaded) < len(chunk):
            st.caption(local_pending)
        st.divider()


def render_batch_progress_footer(*, loaded_total: int, total_files: int, is_running: bool, progress: dict) -> None:
    """Render standard progress footer for sequential sub-batch loading."""
    if is_running:
        st.caption(_msg(
            "progress_running",
            loaded=loaded_total,
            total=total_files,
            done=progress["done"],
            all_batches=progress["total"],
            errors=progress["errors"],
        ))
    elif loaded_total < total_files:
        st.warning(_msg("progress_interrupted", loaded=loaded_total, total=total_files))
    else:
        st.caption(_msg("progress_done", total=total_files))


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
