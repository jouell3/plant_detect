import os

import requests
import streamlit as st
from loguru import logger

st.set_page_config(page_title="Batch Predict", layout="wide")

API_URL = os.environ.get("API_URL", "https://plant-detect-backend-649164185154.europe-west1.run.app")

GRID_COLS = 5
GRID_ROWS = 5
PAGE_SIZE = GRID_COLS * GRID_ROWS  # 25


# ---------------------------------------------------------------------------
# Cached predictions — keyed on image bytes so results survive reruns
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def cached_predict_top3(img_bytes: bytes, filename: str) -> dict:
    """Call /predict_herb and return the full response dict {model: [top3]}."""
    logger.info("predict_herb | file={}", filename)
    response = requests.post(
        f"{API_URL}/predict_herb",
        files={"file": (filename, img_bytes, "image/jpeg")},
        timeout=60,
    )
    response.raise_for_status()
    return response.json()


# ---------------------------------------------------------------------------
# Session state init
# ---------------------------------------------------------------------------
if "predict_image_files" not in st.session_state:
    st.session_state.predict_image_files = []  # list of {"name": str, "bytes": bytes}
if "predict_page" not in st.session_state:
    st.session_state.predict_page = 0
if "predict_uploader_key" not in st.session_state:
    st.session_state.predict_uploader_key = 0

# ---------------------------------------------------------------------------
# Header + file uploader
# ---------------------------------------------------------------------------
st.title("Batch Predictions")
st.markdown("""Cette page vous permet de faire des prédictions sur plusieurs images à la fois. Il suffit de les uploader ci-dessous, puis de naviguer dans les pages de résultats.
            \n Vous pourrez visualiser les prédictions de deux modèles différents : le modèle PyTorch (ResNet18) et un modèle Sklearn utilisant des features extraites d'un backbone EfficientNet B3 de 1536 dimensions.""")
st.markdown("Note : les prédictions sont faites en temps réel via des appels à l'API hébergée sur Google Cloud Run, donc un peu de patience pendant le chargement !")


col_path, col_btn = st.columns([5, 1])
with col_path:
    uploaded_images = st.file_uploader(
        label="Upload images",
        label_visibility="collapsed",
        accept_multiple_files=True,
        type=["jpg", "jpeg", "png"],
        key=f"predict_uploader_{st.session_state.predict_uploader_key}",
    )
with col_btn:
    load_clicked = st.button("Load", use_container_width=True)

if load_clicked:
    if not uploaded_images:
        st.error("Please upload at least one image.")
    else:
        st.session_state.predict_image_files = [
            {"name": f.name, "bytes": f.read()} for f in uploaded_images
        ]
        st.session_state.predict_page = 0
        st.session_state.predict_uploader_key += 1  # reset uploader on next render
        cached_predict_top3.clear()  # clear cached predictions when new images are loaded
        st.success(f"Loaded {len(st.session_state.predict_image_files)} images.")

if not st.session_state.predict_image_files:
    st.stop()

# ---------------------------------------------------------------------------
# Grid
# ---------------------------------------------------------------------------
total_pages = max(1, (len(st.session_state.predict_image_files) + PAGE_SIZE - 1) // PAGE_SIZE)
page = st.session_state.predict_page
start = page * PAGE_SIZE
page_files = st.session_state.predict_image_files[start : start + PAGE_SIZE]

for row in range(GRID_ROWS):
    cols = st.columns(GRID_COLS)
    for col_idx in range(GRID_COLS):
        img_idx = row * GRID_COLS + col_idx
        if img_idx >= len(page_files):
            break
        file = page_files[img_idx]

        with cols[col_idx]:
            st.image(file["bytes"], width=250)
            st.caption(file["name"])

            with st.spinner(""):
                try:
                    data = cached_predict_top3(file["bytes"], file["name"])
                except Exception as e:
                    st.error(f"API error: {e}")
                    continue

            for model_key, top3 in data.items():
                st.markdown(f"**{model_key.upper()}**", unsafe_allow_html=True)
                for rank, pred in enumerate(top3, 1):
                    bar_color = "#2e7d32" if rank == 1 else "#555"
                    st.markdown(
                        f"<div style='display:flex; justify-content:space-between;"
                        f"font-size:0.78rem; margin-bottom:2px;'>"
                        f"<span>{rank}. {pred['species']}</span>"
                        f"<span style='color:{bar_color}; font-weight:bold;'>{pred['confidence']:.1%}</span>"
                        f"</div>",
                        unsafe_allow_html=True,
                    )

# ---------------------------------------------------------------------------
# Pagination
# ---------------------------------------------------------------------------
st.divider()
p_left, p_mid, p_right = st.columns([1, 2, 1])

with p_left:
    if st.button("← Prev", disabled=(page == 0), use_container_width=True):
        st.session_state.predict_page -= 1
        st.rerun()

with p_mid:
    end_img = min(start + PAGE_SIZE, len(st.session_state.predict_image_files))
    st.markdown(
        f"<p style='text-align:center; padding-top:8px;'>"
        f"Page {page + 1} / {total_pages} &nbsp;·&nbsp; images {start + 1}–{end_img}"
        f"</p>",
        unsafe_allow_html=True,
    )

with p_right:
    if st.button("Next →", disabled=(page >= total_pages - 1), use_container_width=True):
        st.session_state.predict_page += 1
        st.rerun()
