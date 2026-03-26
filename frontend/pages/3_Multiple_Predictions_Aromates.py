import csv
import io
import json
import os
import time
from collections import Counter
from pathlib import Path

import requests
import streamlit as st
from loguru import logger

from styles import COLORS, confidence_color
from utils import validate_images_batch, show_validation_errors, show_validation_summary

st.set_page_config(page_title="Batch Predict", layout="wide")

#API_URL = os.environ.get("API_URL", "https://plant-detect-backend-649164185154.europe-west1.run.app")
#API_URL = "http://localhost:8080"
API_URL = "https://herb-predictor-966041648100.europe-west1.run.app"
RETRY_DELAYS_SECONDS = (0.8, 1.6)

GRID_COLS = 5
GRID_ROWS = 5
PAGE_SIZE = GRID_COLS * GRID_ROWS  # 25

MODE_INDIVIDUAL = "Individuel — Top-3"
MODE_BATCH      = "Batch — Top-1"

_FICHES_PATH = Path(__file__).parent.parent / "fiches.json"
FICHES: dict = json.loads(_FICHES_PATH.read_text(encoding="utf-8")) if _FICHES_PATH.exists() else {}


def _normalize_species_key(value: str) -> str:
    return (value or "").strip().lower().replace("-", " ")


def _display_species_name(species: str) -> str:
    fiche = FICHES.get(_normalize_species_key(species), {})
    return fiche.get("nom_fr", species)


# ---------------------------------------------------------------------------
# API helpers
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def cached_predict_top3(img_bytes: bytes, filename: str) -> dict:
    """Call /predict_herb → {model: [{species, confidence}, ...]}."""
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


def fetch_predict_batch(files: list[dict]) -> dict:
    """Call /predict-set → {filename: {model: {species, confidence}}}."""
    logger.info("predict-set | {} files", len(files))
    last_error = None
    for idx, delay in enumerate((0.0, *RETRY_DELAYS_SECONDS)):
        if delay > 0:
            time.sleep(delay)
        try:
            response = requests.post(
                f"{API_URL}/predict-set",
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
            logger.warning("predict-set failed | attempt={} | error={}", idx + 1, e)
    raise last_error


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

def _predictions_table(rows: list[dict]) -> str:
    """Compact HTML table: Modèle | Prédiction | Confiance."""
    header = (
        "<tr style='background:#f5f5f5'>"
        "<th style='text-align:left; padding:3px 6px; font-size:0.90rem'>Modèle</th>"
        "<th style='text-align:left; padding:3px 6px; font-size:0.90rem'>Prédiction</th>"
        "<th style='text-align:right; padding:3px 6px; font-size:0.90rem'>Confiance</th>"
        "</tr>"
    )
    body = ""
    for r in rows:
        color = confidence_color(r["confidence"])
        species_label = _display_species_name(r["species"])
        body += (
            f"<tr>"
            f"<td style='padding:2px 6px; font-size:0.90rem'>{r['model']}</td>"
            f"<td style='padding:2px 6px; font-size:0.90rem'>{species_label}</td>"
            f"<td style='padding:2px 6px; font-size:0.90rem; color:{color}; font-weight:bold; text-align:right'>"
            f"{r['confidence']:.1%}</td>"
            f"</tr>"
        )
    return (
        f"<table style='width:100%; border-collapse:collapse; margin-top:6px; margin-bottom:4px; "
        f"border:1px solid #e0e0e0; border-radius:4px'>"
        f"<thead>{header}</thead><tbody>{body}</tbody></table>"
    )


def _consensus_line(rows: list[dict]) -> str:
    """Top herb with vote count and average confidence of agreeing models."""
    counts = Counter(r["species"] for r in rows)
    top_herb, vote_count = counts.most_common(1)[0]
    total = len(rows)
    avg_conf = sum(r["confidence"] for r in rows if r["species"] == top_herb) / vote_count
    color = confidence_color(avg_conf)
    top_herb_label = _display_species_name(top_herb)
    return (
        f"<div style='font-size:1.3rem; margin-top:4px; text-align:center; width:100%'>"
        f"<b>{top_herb_label}</b> "
        f"<span style='color:#616161'>({vote_count}/{total} modèles)</span>"
        f" — <span style='color:{color}; font-weight:bold'>{avg_conf:.1%} moy.</span>"
        f"</div>"
    )


# ---------------------------------------------------------------------------
# Session state init
# ---------------------------------------------------------------------------
if "predict_image_files" not in st.session_state:
    st.session_state.predict_image_files = []
if "predict_page" not in st.session_state:
    st.session_state.predict_page = 0
if "predict_uploader_key" not in st.session_state:
    st.session_state.predict_uploader_key = 0
if "predict_batch_results" not in st.session_state:
    st.session_state.predict_batch_results = {}   # {filename: {model: {species, confidence}}}

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.title("Batch Predictions")
st.markdown(
    "Cette page vous permet de faire des prédictions sur plusieurs images à la fois. "
    "Il suffit de les sélectionner ci-dessous, puis de naviguer dans les pages de résultats.\n\n"
    "Vous pourrez visualiser les prédictions de quatre modèles différents : "
    "PyTorch (ResNet18), Sklearn (EfficientNet B3), PyTorch Large et TensorFlow."
)

# ---------------------------------------------------------------------------
# Mode selector
# ---------------------------------------------------------------------------
predict_mode = st.radio("Mode de prédiction", [MODE_INDIVIDUAL, MODE_BATCH], horizontal=True)

if predict_mode == MODE_INDIVIDUAL:
    st.info(
        "**Mode individuel** : chaque image est envoyée séparément à l'API. "
        "Vous obtiendrez les **3 meilleures prédictions** par modèle — idéal pour explorer les résultats en détail. "
        "Les résultats s'affichent progressivement au fur et à mesure des appels."
    )
else:
    st.info(
        "**Mode batch** : toutes les images sont envoyées en **une seule requête**. "
        "Vous obtenez uniquement la **meilleure prédiction** par modèle. "
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
        key=f"predict_uploader_{st.session_state.predict_uploader_key}",
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
            st.session_state.predict_image_files = image_files
            st.session_state.predict_page = 0
            st.session_state.predict_uploader_key += 1
            st.session_state.predict_batch_results = {}
            cached_predict_top3.clear()

            if predict_mode == MODE_BATCH:
                with st.spinner(f"Envoi de {len(image_files)} images en batch…"):
                    try:
                        st.session_state.predict_batch_results = fetch_predict_batch(image_files)
                        st.success(f"{len(st.session_state.predict_batch_results)} images traitées.")
                    except Exception as e:
                        st.error(f"Erreur API batch: {e}")
                        logger.error("predict-set error | {}", e)
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
    st.caption("Conseil : en dessous de 50%, reprenez la photo avec un meilleur éclairage et un cadrage plus proche.")

    st.markdown("### Export")
    if st.button("Générer le CSV", use_container_width=True):
        with st.spinner("Génération du CSV en cours..."):
            buf = io.StringIO()
            writer = csv.writer(buf)
            all_files = st.session_state.predict_image_files

            if predict_mode == MODE_BATCH and st.session_state.predict_batch_results:
                sample = next(iter(st.session_state.predict_batch_results.values()), {})
                model_keys = list(sample.keys())
                header = ["filename"]
                for mk in model_keys:
                    header += [f"{mk}_top1", f"{mk}_confidence"]
                header += ["agreement"]
                writer.writerow(header)
                for f in all_files:
                    res = st.session_state.predict_batch_results.get(f["name"])
                    if not res:
                        continue
                    top1s = [res[mk]["species"] for mk in model_keys]
                    row = [f["name"]]
                    for mk in model_keys:
                        row += [res[mk]["species"], f"{res[mk]['confidence']:.4f}"]
                    row.append(len(set(top1s)) == 1)
                    writer.writerow(row)
            else:
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
                success_count = failure_count = 0
                for f in all_files:
                    try:
                        d = cached_predict_top3(f["bytes"], f["name"])
                    except Exception:
                        failure_count += 1
                        continue
                    top1s = [d[mk][0]["species"] for mk in model_keys]
                    row = [f["name"]]
                    for mk in model_keys:
                        row += [d[mk][0]["species"], f"{d[mk][0]['confidence']:.4f}"]
                    row.append(len(set(top1s)) == 1)
                    writer.writerow(row)
                    success_count += 1
                if success_count:
                    st.success(f"CSV prêt : {success_count} image(s) exportée(s).")
                if failure_count:
                    st.warning(f"{failure_count} image(s) ignorée(s) suite à une erreur API.")

            st.download_button(
                "Télécharger",
                buf.getvalue().encode("utf-8"),
                file_name="predictions.csv",
                mime="text/csv",
                use_container_width=True,
            )

# ---------------------------------------------------------------------------
# Grid
# ---------------------------------------------------------------------------
total_pages = max(1, (len(st.session_state.predict_image_files) + PAGE_SIZE - 1) // PAGE_SIZE)
page  = st.session_state.predict_page
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
            # ── Batch mode ─────────────────────────────────────────────
            if predict_mode == MODE_BATCH:
                data = st.session_state.predict_batch_results.get(file["name"])
                if data is None:
                    st.image(file["bytes"], width="stretch")
                    st.caption(file["name"])
                    st.warning("Pas de résultat — rechargez en mode batch.")
                    continue

                rows = [{"model": k, "species": v["species"], "confidence": v["confidence"]} for k, v in data.items()]
                first_conf = rows[0]["confidence"]
                low_confidence = first_conf < min_confidence
                st.image(file["bytes"], width="stretch")
                st.caption(f"{'⚠️ ' if low_confidence else ''}{file['name']}")
                st.markdown(_consensus_line(rows), unsafe_allow_html=True)
                st.markdown(" ")
                st.markdown(_predictions_table(rows), unsafe_allow_html=True)
                
                st.markdown(" ")
                st.markdown(" ")

            # ── Individual mode ─────────────────────────────────────────
            else:
                try:
                    data = cached_predict_top3(file["bytes"], file["name"])
                except Exception as e:
                    st.image(file["bytes"], width="stretch")
                    st.caption(file["name"])
                    st.error(f"Erreur API: {e}")
                    continue

                first_conf = data[list(data.keys())[0]][0]["confidence"]
                low_confidence = first_conf < min_confidence
                st.image(file["bytes"], width="stretch")
                st.caption(f"{'⚠️ ' if low_confidence else ''}{file['name']}")

                # Ligne de consensus sur les top-1 des modèles (comme en mode batch)
                top1_rows = [
                    {"model": model_key, "species": preds[0]["species"], "confidence": preds[0]["confidence"]}
                    for model_key, preds in data.items()
                    if preds
                ]
                if top1_rows:
                    st.markdown(_consensus_line(top1_rows), unsafe_allow_html=True)
                st.markdown(" ")

                for model_key, top3 in data.items():
                    st.markdown(f"**{model_key.upper()}**", unsafe_allow_html=True)
                    for rank, pred in enumerate(top3, 1):
                        color = confidence_color(pred["confidence"]) if rank == 1 else "#999"
                        species_label = _display_species_name(pred["species"])
                        st.markdown(
                            f"<div style='display:flex; justify-content:space-between;"
                            f"font-size:1.1rem; margin-bottom:1px;'>"
                            f"<span>{rank}. {species_label}</span>"
                            f"<span style='color:{color}; font-weight:bold;'>{pred['confidence']:.1%}</span>"
                            f"</div>",
                            unsafe_allow_html=True,
                        )
                st.markdown(" ")
                st.markdown(" ")

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