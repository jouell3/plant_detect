import csv
import io
import os
import time
from pathlib import Path

import requests
import streamlit as st
from loguru import logger

# Local imports for styling and validation
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from styles import COLORS, confidence_color
from utils import validate_images_batch, show_validation_errors, show_validation_summary

st.set_page_config(page_title="Batch Predict", layout="wide")

API_URL = os.environ.get("API_URL", "https://plant-detect-backend-649164185154.europe-west1.run.app")
RETRY_DELAYS_SECONDS = (0.8, 1.6)

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
    last_error = None
    for idx, delay in enumerate((0.0, *RETRY_DELAYS_SECONDS)):
        if delay > 0:
            time.sleep(delay)
        try:
            response = requests.post(
                f"{API_URL}/predict_herb",
                files={"file": (filename, img_bytes, "image/jpeg")},
                timeout=60,
            )
            response.raise_for_status()
            return response.json()
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout, requests.exceptions.HTTPError) as e:
            last_error = e
            logger.warning("predict_herb failed | file={} | attempt={} | error={}", filename, idx + 1, e)
    raise last_error


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
        st.error("Veuillez uploader au moins une image.")
    else:
        # Validate images before processing
        valid_files, invalid_files = validate_images_batch(uploaded_images)
        
        # Show validation results
        show_validation_summary(len(valid_files), len(uploaded_images))
        show_validation_errors(invalid_files)
        
        if valid_files:
            st.session_state.predict_image_files = [
                {"name": f.name, "bytes": f.read()} for f in valid_files
            ]
            st.session_state.predict_page = 0
            st.session_state.predict_uploader_key += 1  # reset uploader on next render
            cached_predict_top3.clear()  # clear cached predictions when new images are loaded
        else:
            st.session_state.predict_image_files = []

if not st.session_state.predict_image_files:
    st.stop()

# ---------------------------------------------------------------------------
# Sidebar — filters + CSV export
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### Filtres")
    min_confidence_pct = st.slider("Confiance minimale", 0, 100, 0, 5, format="%d%%")
    min_confidence = min_confidence_pct / 100
    st.caption("Conseil: en dessous de 50%, reprenez la photo avec un meilleur éclairage et un cadrage plus proche.")

    st.markdown("### Export")
    if st.button("Générer le CSV", use_container_width=True):
        with st.spinner("Génération du CSV en cours..."):
            buf = io.StringIO()
            writer = csv.writer(buf)
            all_files = st.session_state.predict_image_files
            sample_data = None
            for f in all_files:
                try:
                    sample_data = cached_predict_top3(f["bytes"], f["name"])
                    break
                except Exception:
                    pass

            model_keys = list(sample_data.keys()) if sample_data else []
            header = ["filename"]
            for mk in model_keys:
                header += [f"{mk}_top1", f"{mk}_confidence"]
            header += ["agreement"]
            writer.writerow(header)

            success_count = 0
            failure_count = 0
            for f in all_files:
                try:
                    d = cached_predict_top3(f["bytes"], f["name"])
                except Exception:
                    failure_count += 1
                    continue

                row = [f["name"]]
                top1s = []
                for mk in model_keys:
                    top1s.append(d[mk][0]["species"])
                    row += [d[mk][0]["species"], f"{d[mk][0]['confidence']:.4f}"]
                row.append(len(set(top1s)) == 1)
                writer.writerow(row)
                success_count += 1

            st.download_button(
                "Télécharger",
                buf.getvalue().encode("utf-8"),
                file_name="predictions.csv",
                mime="text/csv",
                use_container_width=True,
            )
            if success_count:
                st.success(f"CSV prêt: {success_count} image(s) exportée(s).")
            if failure_count:
                st.warning(f"{failure_count} image(s) ignorée(s) suite à une erreur API.")

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
            try:
                data = cached_predict_top3(file["bytes"], file["name"])
            except Exception as e:
                st.image(file["bytes"], width=None)
                st.caption(file["name"])
                st.error(f"Erreur API: {e}")
                continue

            # Apply confidence filter — dim card if all models are below threshold
            first_conf = data[list(data.keys())[0]][0]["confidence"]
            low_confidence = first_conf < min_confidence

            st.image(file["bytes"], width=350)
            caption = f"{'⚠️ ' if low_confidence else ''}{file['name']}"
            st.caption(caption)

            for model_key, top3 in data.items():
                st.markdown(f"**{model_key.upper()}**", unsafe_allow_html=True)
                for rank, pred in enumerate(top3, 1):
                    color = confidence_color(pred['confidence']) if rank == 1 else "#999"
                    st.markdown(
                        f"<div style='display:flex; justify-content:space-between;"
                        f"font-size:0.78rem; margin-bottom:2px;'>"
                        f"<span>{rank}. {pred['species']}</span>"
                        f"<span style='color:{color}; font-weight:bold;'>{pred['confidence']:.1%}</span>"
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
    st.metric(
        label="Progression",
        value=f"Page {page + 1} / {total_pages}",
        delta=f"images {start + 1}-{end_img}",
    )
    target_page = st.number_input(
        "Aller à la page",
        min_value=1,
        max_value=total_pages,
        value=page + 1,
        step=1,
        key="predict_jump_page",
    )
    if target_page != page + 1:
        st.session_state.predict_page = int(target_page) - 1
        st.rerun()

with p_right:
    if st.button("Next →", disabled=(page >= total_pages - 1), use_container_width=True):
        st.session_state.predict_page += 1
        st.rerun()
