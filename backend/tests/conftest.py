"""
Shared fixtures for the test suite.

All model loading and prediction functions are patched so that tests run
without actual ML weights and without network access to GCS.
"""
import io
import pytest
from PIL import Image


# ---------------------------------------------------------------------------
# Canonical stub return values reused across tests
# ---------------------------------------------------------------------------
HERB_TOP3 = [("Basilic", 0.9), ("Menthe", 0.07), ("Thym", 0.03)]
ILLNESS_TOP3 = [("Healthy", 0.85), ("Powdery_mildew", 0.10), ("Rust", 0.05)]

_HERB_SINGLE = ("Basilic", 0.9)
_ILLNESS_SINGLE = ("Healthy", 0.85)


def _herb_set(paths):
    return [_HERB_SINGLE] * len(paths)


def _illness_set(paths):
    return [_ILLNESS_SINGLE] * len(paths)


# ---------------------------------------------------------------------------
# Autouse fixture: patch all model functions before every test
# ---------------------------------------------------------------------------
@pytest.fixture(autouse=True)
def mock_all_models(monkeypatch):
    """Replace every model function in app.api.main with lightweight stubs."""
    import app.api.main as api_mod

    # Startup load functions — must be no-ops so the lifespan doesn't block
    monkeypatch.setattr(api_mod, "load_model_pytorch", lambda: None)
    monkeypatch.setattr(api_mod, "load_model_sklearn", lambda: None)
    monkeypatch.setattr(api_mod, "load_model_pytorch_large", lambda: None)
    monkeypatch.setattr(api_mod, "load_model_tensorflow", lambda: None)
    monkeypatch.setattr(api_mod, "load_model_illness", lambda: None)

    # Single-image prediction stubs
    monkeypatch.setattr(api_mod, "pt_top3", lambda path: HERB_TOP3)
    monkeypatch.setattr(api_mod, "sk_top3", lambda path: HERB_TOP3)
    monkeypatch.setattr(api_mod, "ptl_top3", lambda path: HERB_TOP3)
    monkeypatch.setattr(api_mod, "tf_top3", lambda path: HERB_TOP3)
    monkeypatch.setattr(api_mod, "illness_top3", lambda path: ILLNESS_TOP3)

    # Batch prediction stubs
    monkeypatch.setattr(api_mod, "pt_set", _herb_set)
    monkeypatch.setattr(api_mod, "sk_set", _herb_set)
    monkeypatch.setattr(api_mod, "ptl_set", _herb_set)
    monkeypatch.setattr(api_mod, "tf_set", _herb_set)
    monkeypatch.setattr(api_mod, "illness_set", _illness_set)


# ---------------------------------------------------------------------------
# Image helper
# ---------------------------------------------------------------------------
def make_image_bytes(suffix: str = ".jpg") -> bytes:
    """Return raw bytes of a tiny 10x10 RGB image."""
    img = Image.new("RGB", (10, 10), color=(100, 150, 200))
    buf = io.BytesIO()
    fmt = "JPEG" if suffix in (".jpg", ".jpeg") else "PNG"
    img.save(buf, format=fmt)
    return buf.getvalue()
