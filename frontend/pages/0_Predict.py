from __future__ import annotations

import os
from io import BytesIO

from PIL import Image
import requests
import streamlit as st

API_URL = os.getenv("PLANT_DETECT_API_URL", "http://localhost:8000")
BACKEND_LABELS = {
    "pytorch": "PyTorch",
    "tensorflow": "TensorFlow 2.15",
}

st.set_page_config(page_title="Predict", layout="centered")
st.title("Plant Detection App")
st.markdown("### Upload an image or use your camera, then choose the inference backend.")


@st.cache_data(ttl=15)
def fetch_model_options() -> dict:
    response = requests.get(f"{API_URL}/model-options", timeout=10)
    response.raise_for_status()
    return response.json()


def open_uploaded_image(uploaded_file) -> Image.Image:
    return Image.open(BytesIO(uploaded_file.getvalue())).convert("RGB")


def selected_image():
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    camera_file = st.camera_input("Or take a picture")

    if uploaded_file is not None:
        return uploaded_file, open_uploaded_image(uploaded_file)

    if camera_file is not None:
        return camera_file, open_uploaded_image(camera_file)

    return None, None


try:
    model_options = fetch_model_options()
    statuses = {
        item["backend"]: item
        for item in model_options.get("backends", [])
    }
except requests.RequestException as exc:
    statuses = {}
    st.warning(f"API unavailable on {API_URL}: {exc}")

available_backends = [
    backend
    for backend, status in statuses.items()
    if status.get("available")
]
default_backend = "pytorch" if "pytorch" in available_backends else (available_backends[0] if available_backends else None)

backend = st.selectbox(
    "Inference backend",
    options=available_backends or list(BACKEND_LABELS),
    index=(available_backends.index(default_backend) if default_backend in available_backends else 0),
    format_func=lambda value: BACKEND_LABELS.get(value, value),
)

status = statuses.get(backend)
if status:
    if status.get("available"):
        st.caption(f"Model loaded from `{status['model_path']}` with {status['num_classes']} classes.")
    else:
        st.error(status.get("error", f"{backend} backend unavailable."))

source_file, preview_image = selected_image()

if preview_image is not None:
    st.image(preview_image, caption="Selected image")

predict_disabled = source_file is None or (status is not None and not status.get("available"))

if st.button("Run prediction", use_container_width=True, disabled=predict_disabled):
    try:
        with st.spinner(f"Running inference with {BACKEND_LABELS.get(backend, backend)}..."):
            response = requests.post(
                f"{API_URL}/predict_herb",
                files={"file": (source_file.name, source_file.getvalue(), source_file.type or "image/jpeg")},
                data={"backend": backend},
                timeout=60,
            )
    except requests.RequestException as exc:
        st.error(f"Prediction request failed: {exc}")
        st.stop()

    if response.ok:
        payload = response.json()
        predictions = payload.get("predictions", [])
        if predictions:
            top_prediction = predictions[0]
            st.success(
                f"Top prediction: {top_prediction['species']} "
                f"({top_prediction['confidence']:.2%})"
            )
            st.markdown(
                f"#### Top predictions with {BACKEND_LABELS.get(payload.get('backend', backend), backend)}"
            )
            for prediction in predictions:
                st.write(f"{prediction['species']}: {prediction['confidence']:.2%}")
                st.progress(float(prediction["confidence"]))
        else:
            st.warning("The API returned no predictions.")
    else:
        try:
            error_payload = response.json()
            detail = error_payload.get("detail", response.text)
        except ValueError:
            detail = response.text
        st.error(f"Prediction failed: {detail}")
