import io
import os

import requests
import streamlit as st
from loguru import logger
from PIL import Image

API_URL = os.environ.get("API_URL", "https://herb-predictor-966041648100.europe-west1.run.app")

st.set_page_config(page_title="Plant Predictor", layout="wide")

st.title("🌿 Plant Predictor")
st.caption("Upload one or several images — or take a photo — to identify your aromatic plants.")

# ---------------------------------------------------------------------------
# Input
# ---------------------------------------------------------------------------
tab_upload, tab_camera = st.tabs(["📁 Upload images", "📷 Camera"])

uploaded_files = []

with tab_upload:
    files = st.file_uploader(
        "Choose one or several images",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )
    if files:
        uploaded_files = files

with tab_camera:
    photo = st.camera_input("Take a picture", label_visibility="collapsed")
    if photo:
        uploaded_files = [photo]

# ---------------------------------------------------------------------------
# Predict
# ---------------------------------------------------------------------------
if not uploaded_files:
    st.stop()

st.divider()

if st.button("🔍 Identify", type="primary", use_container_width=False):
    is_batch = len(uploaded_files) > 1

    with st.spinner("Analysing…"):
        try:
            if is_batch:
                logger.info("predict-set | {} images", len(uploaded_files))
                response = requests.post(
                    f"{API_URL}/predict-set",
                    files=[
                        ("files", (f.name, f.getvalue(), f.type or "image/jpeg"))
                        for f in uploaded_files
                    ],
                    timeout=60,
                )
            else:
                f = uploaded_files[0]
                logger.info("predict_herb | file={}", f.name)
                response = requests.post(
                    f"{API_URL}/predict_herb",
                    files={"file": (f.name, f.getvalue(), f.type or "image/jpeg")},
                    timeout=60,
                )
            response.raise_for_status()

        except requests.exceptions.ConnectionError:
            st.error("Cannot reach the API. Check that the service is running.")
            logger.error("API connection error | url={}", API_URL)
            st.stop()
        except requests.exceptions.HTTPError as e:
            st.error(f"API error: {e}")
            logger.error("API HTTP error | {}", e)
            st.stop()

    # ── Parse results ─────────────────────────────────────────────────────
    if is_batch:
        raw = response.json()   # list of {filename, species, confidence}
        results = {item["filename"]: item for item in raw}
    else:
        data = response.json()  # {predictions: [{species, confidence}, ...]}
        results = {uploaded_files[0].name: {
            "filename": uploaded_files[0].name,
            "species": data["predictions"][0]["species"],
            "confidence": data["predictions"][0]["confidence"],
            "top3": data["predictions"],
        }}

    # ── Display ───────────────────────────────────────────────────────────
    st.subheader("Results")
    cols = st.columns(min(len(uploaded_files), 4))

    for idx, f in enumerate(uploaded_files):
        col = cols[idx % len(cols)]
        res = results.get(f.name, {})

        with col:
            img = Image.open(io.BytesIO(f.getvalue()))
            st.image(img, width=180)

            species    = res.get("species", "—")
            confidence = res.get("confidence", 0)

            color = "#2e7d32" if confidence >= 0.75 else "#f57c00" if confidence >= 0.5 else "#c62828"
            st.markdown(
                f"<p style='text-align:center; font-size:1.1rem; font-weight:600; margin:4px 0'>"
                f"{species}</p>"
                f"<p style='text-align:center; color:{color}; font-size:0.95rem; margin:0'>"
                f"{confidence:.0%} confidence</p>",
                unsafe_allow_html=True,
            )

            # Top-3 only available on single-image endpoint
            if "top3" in res:
                with st.expander("Top 3"):
                    for rank, pred in enumerate(res["top3"], 1):
                        bar_pct = int(pred["confidence"] * 100)
                        st.markdown(
                            f"**{rank}. {pred['species']}** — {pred['confidence']:.0%}"
                        )
                        st.progress(bar_pct)
