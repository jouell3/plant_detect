import csv
import io
import json
import os
import threading
import time
from collections import Counter
from pathlib import Path

import requests
import streamlit as st
from loguru import logger

from styles import COLORS, confidence_color
from utils import (
    clear_batch_session_tracking,
    get_batch_bg_state,
    get_streamlit_session_id,
    reset_batch_page_state,
    render_batch_lot_grids,
    render_batch_progress_footer,
    run_sequential_subbatch_fetch,
    show_validation_errors,
    show_validation_summary,
    post_with_retries,
    validate_images_batch,
)

st.set_page_config(page_title="Batch Predict", layout="wide")

API_URL = os.environ.get("API_URL", "https://plant-detect-backend-649164185154.europe-west1.run.app")
#API_URL = "http://localhost:8080"
#API_URL = "https://herb-predictor-966041648100.europe-west1.run.app"
RETRY_DELAYS_SECONDS = (0.8, 1.6)

GRID_COLS = 5
GRID_ROWS = 4
PAGE_SIZE = GRID_COLS * GRID_ROWS  # 20

MODE_INDIVIDUAL = "Individuel — Top-3"
MODE_BATCH      = "Batch — Top-1"

_FICHES_PATH = Path(__file__).parent.parent / "fiches.json"
FICHES: dict = json.loads(_FICHES_PATH.read_text(encoding="utf-8")) if _FICHES_PATH.exists() else {}

# ---------------------------------------------------------------------------
# Background fetch infrastructure (shared utils)
# ---------------------------------------------------------------------------
_BG_STATE = get_batch_bg_state("aromates")


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
    response = post_with_retries(
        url=f"{API_URL}/predict_herb",
        files={"file": (filename, img_bytes, "image/jpeg")},
        timeout=60,
        retry_delays_seconds=RETRY_DELAYS_SECONDS,
        log_message=f"predict_herb failed | file={filename}",
    )
    return response.json()


def fetch_predict_batch(files: list[dict]) -> dict:
    """Call /predict-set → {filename: {model: {species, confidence}}}."""
    logger.info("predict-set | {} files", len(files))
    response = post_with_retries(
        url=f"{API_URL}/predict-set",
        files=[("files", (f["name"], f["bytes"], "image/jpeg")) for f in files],
        timeout=120,
        retry_delays_seconds=RETRY_DELAYS_SECONDS,
        log_message="predict-set failed",
    )
    results = response.json()
    return {
        item["filename"]: {k: v for k, v in item.items() if k != "filename"}
        for item in results
    }


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

_CELL = "padding:2px 3px; font-size:clamp(0.5rem, 0.95vw, 0.82rem); white-space:nowrap; overflow:hidden; text-overflow:ellipsis; max-width:0"
_HEAD = f"background:#f5f5f5; {_CELL}"

def _predictions_table(rows: list[dict], consensus_species: str | None = None) -> str:
    """Compact HTML table: Modèle | Prédiction | Confiance. Highlights disagreeing rows."""
    header = (
        "<tr>"
        f"<th style='text-align:left; {_HEAD}'>Modèle</th>"
        f"<th style='text-align:left; {_HEAD}'>Prédiction</th>"
        f"<th style='text-align:right; {_HEAD}'>Confiance</th>"
        "</tr>"
    )
    body = ""
    for r in rows:
        color = confidence_color(r["confidence"])
        species_label = _display_species_name(r["species"])
        disagrees = consensus_species is not None and r["species"] != consensus_species
        row_bg = "background-color:#fff3e0;" if disagrees else ""
        body += (
            f"<tr style='{row_bg}'>"
            f"<td style='text-align:left; {_CELL}'>{r['model']}</td>"
            f"<td style='text-align:left; {_CELL}'>{species_label}</td>"
            f"<td style='text-align:right; {_CELL}; color:{color}; font-weight:bold'>{r['confidence']:.1%}</td>"
            f"</tr>"
        )
    return (
        f"<table style='width:100%; table-layout:fixed; border-collapse:collapse; margin-top:6px; margin-bottom:4px; "
        f"border:1px solid #e0e0e0; border-radius:4px; overflow:hidden'>"
        f"<colgroup><col style='width:30%'><col style='width:42%'><col style='width:28%'></colgroup>"
        f"<thead>{header}</thead><tbody>{body}</tbody></table>"
    )


def _consensus_line(rows: list[dict], low_confidence: bool = False, disagreement: bool = False) -> str:
    """Top herb with vote count and average confidence of agreeing models."""
    counts = Counter(r["species"] for r in rows)
    top_herb, vote_count = counts.most_common(1)[0]
    total = len(rows)
    avg_conf = sum(r["confidence"] for r in rows if r["species"] == top_herb) / vote_count
    color = confidence_color(avg_conf)
    top_herb_label = _display_species_name(top_herb)
    conf_icon = "⚠️ " if low_confidence else ""
    split_icon = " 🔀" if disagreement else ""
    return (
        f"<div style='text-align:center; width:100%; margin-top:6px; margin-bottom:2px; line-height:1.4'>"
        f"<div style='font-size:clamp(0.85rem, 1.6vw, 1.3rem); font-weight:bold'>{conf_icon}{top_herb_label}</div>"
        f"<div style='font-size:clamp(0.6rem, 1.0vw, 0.85rem); color:#757575'>({vote_count}/{total} modèles){split_icon}</div>"
        f"<div style='font-size:clamp(0.7rem, 1.2vw, 1.0rem); color:{color}; font-weight:bold'>{avg_conf:.1%} certitude</div>"
        f"</div>"
    )


def _render_batch_grid(files: list[dict], batch_results: dict, min_confidence: float) -> None:
    for row_idx in range(0, len(files), GRID_COLS):
        cols = st.columns(GRID_COLS)
        for col_idx, file in enumerate(files[row_idx : row_idx + GRID_COLS]):
            with cols[col_idx]:
                data = batch_results[file["name"]]
                rows = [{"model": k, "species": v["species"], "confidence": v["confidence"]} for k, v in data.items()]
                disagreement = len(set(r["species"] for r in rows)) > 1
                consensus_species = Counter(r["species"] for r in rows).most_common(1)[0][0]
                low_confidence = any(r["confidence"] < min_confidence for r in rows if r["species"] == consensus_species)
                st.image(file["bytes"], width="stretch")
                st.caption(file["name"])
                st.markdown(_consensus_line(rows, low_confidence=low_confidence, disagreement=disagreement), unsafe_allow_html=True)
                with st.expander("Voir les détails"):
                    st.markdown(_predictions_table(rows, consensus_species=consensus_species), unsafe_allow_html=True)


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
if "predict_batches_loaded" not in st.session_state:
    st.session_state.predict_batches_loaded = set()  # set of page indices already fetched
if "predict_last_mode" not in st.session_state:
    st.session_state.predict_last_mode = None
if "predict_last_uploader_filenames" not in st.session_state:
    st.session_state.predict_last_uploader_filenames = set()

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.title("Predictions en lot d'aromates")
st.markdown("""
    - Cette page vous permet de faire des prédictions sur plusieurs images à la fois.   
    - Il suffit de les sélectionner ci-dessous, puis de naviguer dans les pages de résultats.  
    - Vous pourrez visualiser les prédictions de quatre modèles différents :
      - PyTorch (ResNet18), Sklearn (EfficientNet B3), PyTorch Large et TensorFlow.
    - Si jamais une des prédictions est sous la barre des 60% de certitude, un petit pictogramme apparaîtra au niveau du nom de l'aromate.   
      - Il est possible de choisir le niveau de certitude minimal voulu pour la ligne de concensus dans la barre latérale à droite."""
)

# ---------------------------------------------------------------------------
# Mode selector
# ---------------------------------------------------------------------------
st.markdown("### Choisissez le mode de prédiction")
predict_mode = st.radio(
    "Mode de prédiction",
    [MODE_INDIVIDUAL, MODE_BATCH],
    horizontal=True,
    label_visibility="collapsed",
)

_sid = get_streamlit_session_id()

if st.session_state.predict_last_mode is not None and predict_mode != st.session_state.predict_last_mode:
    reset_batch_page_state(
        session_id=_sid,
        bg_state=_BG_STATE,
        image_files_key="predict_image_files",
        batch_results_key="predict_batch_results",
        batches_loaded_key="predict_batches_loaded",
        page_key="predict_page",
        cache_clear_fn=cached_predict_top3.clear,
    )
st.session_state.predict_last_mode = predict_mode

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

current_uploader_filenames = {f.name for f in uploaded_images} if uploaded_images else set()
loaded_filenames = {f["name"] for f in st.session_state.predict_image_files}
if current_uploader_filenames and current_uploader_filenames != loaded_filenames:
    st.info(f"**{len(uploaded_images)} image(s) sélectionnée(s)**. Cliquez sur **Load** pour lancer cette sélection.")
elif st.session_state.predict_image_files:
    st.caption(f"{len(st.session_state.predict_image_files)} image(s) chargée(s).")
else:
    st.caption("0 image(s) sélectionnée(s). Cliquez sur Load pour lancer les prédictions.")

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
            st.session_state.predict_batches_loaded = set()
            cached_predict_top3.clear()
            clear_batch_session_tracking(_BG_STATE, _sid)

            if predict_mode == MODE_BATCH:
                first_page = image_files[:PAGE_SIZE]
                first_page_ok = False
                with st.spinner(f"Chargement de la page 1 ({len(first_page)} images)…"):
                    try:
                        results = fetch_predict_batch(first_page)
                        st.session_state.predict_batch_results.update(results)
                        st.session_state.predict_batches_loaded.add(0)
                        st.success(f"Page 1 chargée — {len(results)} images.")
                        first_page_ok = True
                    except Exception as e:
                        st.error(f"Erreur API batch: {e}")
                        logger.error("predict-set error | {}", e)

                remaining_files = image_files[PAGE_SIZE:] if first_page_ok else image_files
                if remaining_files:
                    with _BG_STATE["lock"]:
                        _BG_STATE["running"].add(_sid)
                    threading.Thread(
                        target=run_sequential_subbatch_fetch,
                        args=(_sid, remaining_files, PAGE_SIZE, fetch_predict_batch, _BG_STATE, "bg sub-batch fetch failed"),
                        daemon=True,
                    ).start()

if not st.session_state.predict_image_files:
    st.stop()

# ---------------------------------------------------------------------------
# Sidebar — filters + CSV export
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### Filtres")
    min_confidence_pct = st.slider("Confiance minimale", 60, 100, 60, 5, format="%d%%")
    min_confidence = min_confidence_pct / 100
    st.caption("Conseil : en dessous de 60%, reprenez la photo avec un meilleur éclairage et un cadrage plus proche.")

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
# Grid — Batch mode: infinite scroll by chunks of PAGE_SIZE
#         Individual mode: paginated
# ---------------------------------------------------------------------------
all_files = st.session_state.predict_image_files
total_files = len(all_files)

if predict_mode == MODE_BATCH:
    # Collect results that are being accumulated in the background
    with _BG_STATE["lock"]:
        st.session_state.predict_batch_results.update(_BG_STATE["results"].get(_sid, {}))
        is_running = _sid in _BG_STATE["running"]
        progress = _BG_STATE["progress"].get(_sid, {"done": 0, "total": 0, "errors": 0})
        failed_files = list(_BG_STATE["failed_files"].get(_sid, []))

    loaded_total = len(st.session_state.predict_batch_results)

    def _render_aromate_item(file: dict, data: dict) -> None:
        rows = [{"model": k, "species": v["species"], "confidence": v["confidence"]} for k, v in data.items()]
        disagreement = len(set(r["species"] for r in rows)) > 1
        consensus_species = Counter(r["species"] for r in rows).most_common(1)[0][0]
        low_confidence = any(r["confidence"] < min_confidence for r in rows if r["species"] == consensus_species)
        st.image(file["bytes"], width="stretch")
        st.caption(file["name"])
        st.markdown(_consensus_line(rows, low_confidence=low_confidence, disagreement=disagreement), unsafe_allow_html=True)
        with st.expander("Voir les détails"):
            st.markdown(_predictions_table(rows, consensus_species=consensus_species), unsafe_allow_html=True)

    render_batch_lot_grids(
        all_files=all_files,
        batch_results=st.session_state.predict_batch_results,
        page_size=PAGE_SIZE,
        grid_cols=GRID_COLS,
        render_item_fn=_render_aromate_item,
    )
    render_batch_progress_footer(
        loaded_total=loaded_total,
        total_files=total_files,
        is_running=is_running,
        progress=progress,
    )

    if not is_running and failed_files:
        st.info(f"{len(failed_files)} image(s) en échec peuvent être relancées sans perdre les résultats déjà reçus.")
        if st.button("Reprendre les lots échoués", use_container_width=True, key="retry_failed_aromates"):
            with _BG_STATE["lock"]:
                _BG_STATE["running"].add(_sid)
            threading.Thread(
                target=run_sequential_subbatch_fetch,
                args=(_sid, failed_files, PAGE_SIZE, fetch_predict_batch, _BG_STATE, "bg sub-batch retry failed"),
                daemon=True,
            ).start()
            st.rerun()

    if is_running:
        time.sleep(0.7)
        st.rerun()

else:
    # ── Individual mode: paginated ──────────────────────────────────────
    total_pages = max(1, (total_files + PAGE_SIZE - 1) // PAGE_SIZE)
    page = st.session_state.predict_page
    start = page * PAGE_SIZE
    page_files = all_files[start : start + PAGE_SIZE]

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
                    st.image(file["bytes"], width="stretch")
                    st.caption(file["name"])
                    st.error(f"Erreur API: {e}")
                    continue

                top1_rows = [
                    {"model": model_key, "species": preds[0]["species"], "confidence": preds[0]["confidence"]}
                    for model_key, preds in data.items() if preds
                ]
                low_confidence = any(r["confidence"] < min_confidence for r in top1_rows)
                disagreement = len(set(r["species"] for r in top1_rows)) > 1
                consensus_species = Counter(r["species"] for r in top1_rows).most_common(1)[0][0] if top1_rows else None
                st.image(file["bytes"], width="stretch")
                st.caption(file["name"])
                if top1_rows:
                    st.markdown(_consensus_line(top1_rows, low_confidence=low_confidence, disagreement=disagreement), unsafe_allow_html=True)
                with st.expander("Voir les détails"):
                    for model_key, top3 in data.items():
                        st.markdown(f"**{model_key.upper()}**", unsafe_allow_html=True)
                        for rank, pred in enumerate(top3, 1):
                            is_consensus = pred["species"] == consensus_species
                            color = confidence_color(pred["confidence"]) if rank == 1 else "#999"
                            species_label = _display_species_name(pred["species"])
                            bg = "background-color:#e8f5e9; border-radius:3px; padding:1px 4px;" if is_consensus else "padding:1px 4px;"
                            st.markdown(
                                f"<div style='display:flex; justify-content:space-between; {bg}"
                                f"font-size:clamp(0.6rem, 1.25vw, 1.1rem); margin-bottom:1px;'>"
                                f"<span>{rank}. {species_label}</span>"
                                f"<span style='color:{color}; font-weight:bold;'>{pred['confidence']:.1%}</span>"
                                f"</div>",
                                unsafe_allow_html=True,
                            )
                st.markdown(" ")

    # Pagination (individual mode only)
    st.divider()
    p_left, p_mid, p_right = st.columns([1, 2, 1])
    with p_left:
        if st.button("← Prev", disabled=(page == 0), use_container_width=True):
            st.session_state.predict_page -= 1
            st.rerun()
    with p_mid:
        end_img = min(start + PAGE_SIZE, total_files)
        st.metric("Progression", f"Page {page + 1} / {total_pages}", delta=f"images {start + 1}–{end_img}")
        target_page = st.number_input("Aller à la page", min_value=1, max_value=total_pages,
                                      value=page + 1, step=1, key="predict_jump_page")
        if target_page != page + 1:
            st.session_state.predict_page = int(target_page) - 1
            st.rerun()
    with p_right:
        if st.button("Next →", disabled=(page >= total_pages - 1), use_container_width=True):
            st.session_state.predict_page += 1
            st.rerun()