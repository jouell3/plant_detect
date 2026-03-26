import csv
import io
import json
import os
import time
from pathlib import Path

import requests
import streamlit as st
from loguru import logger

from styles import COLORS, confidence_color
from utils import validate_images_batch, show_validation_errors, show_validation_summary

st.set_page_config(page_title="Batch Predict — Maladies", layout="wide")

#API_URL = os.environ.get("API_URL", "https://plant-detect-backend-649164185154.europe-west1.run.app")
#API_URL = "http://localhost:8080"
API_URL = "https://herb-predictor-966041648100.europe-west1.run.app"
RETRY_DELAYS_SECONDS = (0.8, 1.6)

GRID_COLS = 5
GRID_ROWS = 5
PAGE_SIZE = GRID_COLS * GRID_ROWS  # 25

MODE_INDIVIDUAL = "Individuel — Top-3"
MODE_BATCH      = "Batch — Top-1"

_FICHES_ILL_PATH = Path(__file__).parent.parent / "fiches_ill.json"
FICHES_ILL: dict = json.loads(_FICHES_ILL_PATH.read_text(encoding="utf-8")) if _FICHES_ILL_PATH.exists() else {}


# ---------------------------------------------------------------------------
# API helpers
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def cached_predict_illness_top3(img_bytes: bytes, filename: str) -> list:
    """Call /predict_illness → [{illness, confidence}, ...] (top-3)."""
    logger.info("predict_illness | file={}", filename)
    last_error = None
    for idx, delay in enumerate((0.0, *RETRY_DELAYS_SECONDS)):
        if delay > 0:
            time.sleep(delay)
        try:
            response = requests.post(
                f"{API_URL}/predict_illness",
                files={"file": (filename, img_bytes, "image/jpeg")},
                timeout=60,
            )
            response.raise_for_status()
            return response.json()
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout, requests.exceptions.HTTPError) as e:
            last_error = e
            logger.warning("predict_illness failed | file={} | attempt={} | error={}", filename, idx + 1, e)
    raise last_error


def fetch_illness_batch(files: list[dict]) -> dict:
    """Call /predict-set_illness → {filename: {illness, confidence}}."""
    logger.info("predict-set_illness | {} files", len(files))
    last_error = None
    for idx, delay in enumerate((0.0, *RETRY_DELAYS_SECONDS)):
        if delay > 0:
            time.sleep(delay)
        try:
            response = requests.post(
                f"{API_URL}/predict-set_illness",
                files=[("files", (f["name"], f["bytes"], "image/jpeg")) for f in files],
                timeout=120,
            )
            response.raise_for_status()
            results = response.json()
            return {
                item["filename"]: {k: v for k, v in item.items() if k != "filename"}
                for item in results
            }
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout, requests.exceptions.HTTPError) as e:
            last_error = e
            logger.warning("predict-set_illness failed | attempt={} | error={}", idx + 1, e)
    raise last_error


# ---------------------------------------------------------------------------
# Session state init
# ---------------------------------------------------------------------------
if "ill_predict_image_files" not in st.session_state:
    st.session_state.ill_predict_image_files = []
if "ill_predict_page" not in st.session_state:
    st.session_state.ill_predict_page = 0
if "ill_predict_uploader_key" not in st.session_state:
    st.session_state.ill_predict_uploader_key = 0
if "ill_predict_batch_results" not in st.session_state:
    st.session_state.ill_predict_batch_results = {}  # {filename: {illness, confidence}}

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.title("Batch Predictions — Maladies")
st.markdown(
    "Cette page vous permet de faire des prédictions sur plusieurs images à la fois. "
    "Il suffit de les sélectionner ci-dessous, puis de naviguer dans les pages de résultats.\n\n"
    "Le modèle de détection des maladies est basé sur un ResNet18 entraîné sur des images de plantes malades."
)

# ---------------------------------------------------------------------------
# Mode selector
# ---------------------------------------------------------------------------
predict_mode = st.radio("Mode de prédiction", [MODE_INDIVIDUAL, MODE_BATCH], horizontal=True)

if predict_mode == MODE_INDIVIDUAL:
    st.info(
        "**Mode individuel** : chaque image est envoyée séparément à l'API (`/predict_illness`). "
        "Vous obtenez les **3 meilleures prédictions** — idéal pour explorer les résultats en détail. "
        "Les résultats s'affichent progressivement au fur et à mesure des appels."
    )
else:
    st.info(
        "**Mode batch** : toutes les images sont envoyées en **une seule requête** (`/predict-set_illness`). "
        "Vous obtenez uniquement la **meilleure prédiction** par image. "
        "Plus rapide et plus efficace pour traiter un grand nombre d'images d'un coup."
    )

# ---------------------------------------------------------------------------
# File uploader + Load button
# ---------------------------------------------------------------------------
col_path, col_btn = st.columns([5, 1])
with col_path:
    uploaded_images = st.file_uploader(
        label="Upload images",
        label_visibility="collapsed",
        accept_multiple_files=True,
        type=["jpg", "jpeg", "png"],
        key=f"ill_predict_uploader_{st.session_state.ill_predict_uploader_key}",
    )
with col_btn:
    load_clicked = st.button("Load", use_container_width=True)

if load_clicked:
    if not uploaded_images:
        st.error("Veuillez uploader au moins une image.")
    else:
        valid_files, invalid_files = validate_images_batch(uploaded_images)
        show_validation_summary(len(valid_files), len(uploaded_images))
        show_validation_errors(invalid_files)

        if valid_files:
            image_files = [{"name": f.name, "bytes": f.read()} for f in valid_files]
            st.session_state.ill_predict_image_files = image_files
            st.session_state.ill_predict_page = 0
            st.session_state.ill_predict_uploader_key += 1
            st.session_state.ill_predict_batch_results = {}
            cached_predict_illness_top3.clear()

            if predict_mode == MODE_BATCH:
                with st.spinner(f"Envoi de {len(image_files)} images en batch…"):
                    try:
                        st.session_state.ill_predict_batch_results = fetch_illness_batch(image_files)
                        st.success(f"{len(st.session_state.ill_predict_batch_results)} images traitées.")
                    except Exception as e:
                        st.error(f"Erreur API batch: {e}")
                        logger.error("predict-set_illness error | {}", e)
        else:
            st.session_state.ill_predict_image_files = []

if not st.session_state.ill_predict_image_files:
    st.stop()

# ---------------------------------------------------------------------------
# Sidebar — filters + CSV export
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### Filtres")
    min_confidence_pct = st.slider("Confiance minimale", 0, 100, 0, 5, format="%d%%")
    min_confidence = min_confidence_pct / 100
    st.caption("Conseil : en dessous de 50%, reprenez la photo avec un meilleur éclairage et un cadrage plus proche.")

    st.markdown("### Export")
    if st.button("Générer le CSV", use_container_width=True):
        with st.spinner("Génération du CSV en cours..."):
            buf = io.StringIO()
            writer = csv.writer(buf)
            all_files = st.session_state.ill_predict_image_files

            if predict_mode == MODE_BATCH and st.session_state.ill_predict_batch_results:
                writer.writerow(["filename", "illness", "confidence"])
                for f in all_files:
                    res = st.session_state.ill_predict_batch_results.get(f["name"])
                    if not res:
                        continue
                    # res may be {illness, confidence} directly or wrapped in a model key
                    if "illness" in res:
                        writer.writerow([f["name"], res["illness"], f"{res['confidence']:.4f}"])
                    else:
                        pred = next(iter(res.values()))
                        writer.writerow([f["name"], pred["illness"], f"{pred['confidence']:.4f}"])
            else:
                writer.writerow(["filename", "top1_illness", "top1_confidence", "top2_illness", "top2_confidence", "top3_illness", "top3_confidence"])
                success_count = failure_count = 0
                for f in all_files:
                    try:
                        top3 = cached_predict_illness_top3(f["bytes"], f["name"])
                    except Exception:
                        failure_count += 1
                        continue
                    # Flatten: top3 may be a list or wrapped in a model key
                    preds = top3 if isinstance(top3, list) else next(iter(top3.values()))
                    row = [f["name"]]
                    for p in preds[:3]:
                        row += [p["illness"], f"{p['confidence']:.4f}"]
                    writer.writerow(row)
                    success_count += 1
                if success_count:
                    st.success(f"CSV prêt : {success_count} image(s) exportée(s).")
                if failure_count:
                    st.warning(f"{failure_count} image(s) ignorée(s) suite à une erreur API.")

            st.download_button(
                "Télécharger",
                buf.getvalue().encode("utf-8"),
                file_name="predictions_maladies.csv",
                mime="text/csv",
                use_container_width=True,
            )

# ---------------------------------------------------------------------------
# Grid
# ---------------------------------------------------------------------------
total_pages = max(1, (len(st.session_state.ill_predict_image_files) + PAGE_SIZE - 1) // PAGE_SIZE)
page  = st.session_state.ill_predict_page
start = page * PAGE_SIZE
page_files = st.session_state.ill_predict_image_files[start : start + PAGE_SIZE]

for row in range(GRID_ROWS):
    cols = st.columns(GRID_COLS)
    for col_idx in range(GRID_COLS):
        img_idx = row * GRID_COLS + col_idx
        if img_idx >= len(page_files):
            break
        file = page_files[img_idx]

        with cols[col_idx]:
            # ── Batch mode ─────────────────────────────────────────────
            if predict_mode == MODE_BATCH:
                res = st.session_state.ill_predict_batch_results.get(file["name"])
                if res is None:
                    st.image(file["bytes"], width="stretch")
                    st.caption(file["name"])
                    st.warning("Pas de résultat — rechargez en mode batch.")
                    continue

                # Unwrap: res may be {illness, confidence} or {model: {illness, confidence}}
                pred = res if "illness" in res else next(iter(res.values()))
                low_confidence = pred["confidence"] < min_confidence
                color = confidence_color(pred["confidence"])

                fiche = FICHES_ILL.get(pred["illness"], {})
                display_name = fiche.get("nom_maladie_fr") or pred["illness"]

                st.image(file["bytes"], width="stretch")
                st.caption(f"{'⚠️ ' if low_confidence else ''}{file['name']}")
                st.markdown(
                    f"<div style='font-size:1.32rem; margin-bottom:24px'>"
                    f"<b>{display_name}</b> — "
                    f"<span style='color:{color}; font-weight:bold'>{pred['confidence']:.1%}</span>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

            # ── Individual mode ─────────────────────────────────────────
            else:
                try:
                    result = cached_predict_illness_top3(file["bytes"], file["name"])
                except Exception as e:
                    st.image(file["bytes"], width="stretch")
                    st.caption(file["name"])
                    st.error(f"Erreur API: {e}")
                    continue

                # Unwrap: result may be a list or wrapped in a model key
                top3 = result if isinstance(result, list) else next(iter(result.values()))
                first_conf = top3[0]["confidence"]
                low_confidence = first_conf < min_confidence

                st.image(file["bytes"], width="stretch")
                st.caption(f"{'⚠️ ' if low_confidence else ''}{file['name']}")
                for rank, pred in enumerate(top3, 1):
                    color = confidence_color(pred["confidence"]) if rank == 1 else "#999"
                    name = FICHES_ILL.get(pred["illness"], {}).get("nom_maladie_fr") or pred["illness"]
                    st.markdown(
                        f"<div style='display:flex; justify-content:space-between;"
                        f"font-size:1rem; margin-bottom:2px;'>"
                        f"<span>{rank}. {name}</span>"
                        f"<span style='color:{color}; font-weight:bold;'>{pred['confidence']:.1%}</span>"
                        f"</div>",
                        unsafe_allow_html=True,
                    )
                st.markdown(" ")

# ---------------------------------------------------------------------------
# Pagination
# ---------------------------------------------------------------------------
st.divider()
p_left, p_mid, p_right = st.columns([1, 2, 1])

with p_left:
    if st.button("← Prev", disabled=(page == 0), use_container_width=True):
        st.session_state.ill_predict_page -= 1
        st.rerun()

with p_mid:
    end_img = min(start + PAGE_SIZE, len(st.session_state.ill_predict_image_files))
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
        key="ill_predict_jump_page",
    )
    if target_page != page + 1:
        st.session_state.ill_predict_page = int(target_page) - 1
        st.rerun()

with p_right:
    if st.button("Next →", disabled=(page >= total_pages - 1), use_container_width=True):
        st.session_state.ill_predict_page += 1
        st.rerun()