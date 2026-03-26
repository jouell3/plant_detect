import csv
import io
import json
import os
import threading
import time
from pathlib import Path

import requests
import streamlit as st
from loguru import logger

from styles import COLORS, confidence_color
from utils import (
    clear_batch_session_tracking,
    get_batch_bg_state,
    get_streamlit_session_id,
    render_batch_lot_grids,
    render_batch_progress_footer,
    reset_batch_page_state,
    run_sequential_subbatch_fetch,
    show_validation_errors,
    show_validation_summary,
    post_with_retries,
    validate_images_batch,
)

st.set_page_config(page_title="Batch Predict — Maladies", layout="wide")

API_URL = os.environ.get("API_URL", "https://plant-detect-backend-649164185154.europe-west1.run.app")
#API_URL = "http://localhost:8080"
#API_URL = "https://herb-predictor-966041648100.europe-west1.run.app"
RETRY_DELAYS_SECONDS = (0.8, 1.6)

GRID_COLS = 5
GRID_ROWS = 4
PAGE_SIZE = GRID_COLS * GRID_ROWS  # 20

MODE_INDIVIDUAL = "Individuel — Top-3"
MODE_BATCH      = "Batch — Top-1"

_FICHES_ILL_PATH = Path(__file__).parent.parent / "fiches_ill.json"
FICHES_ILL: dict = json.loads(_FICHES_ILL_PATH.read_text(encoding="utf-8")) if _FICHES_ILL_PATH.exists() else {}

# ---------------------------------------------------------------------------
# Background fetch infrastructure (shared utils)
# ---------------------------------------------------------------------------
_BG_STATE = get_batch_bg_state("maladie")


# ---------------------------------------------------------------------------
# API helpers
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def cached_predict_illness_top3(img_bytes: bytes, filename: str) -> list:
    """Call /predict_illness → [{illness, confidence}, ...] (top-3)."""
    logger.info("predict_illness | file={}", filename)
    response = post_with_retries(
        url=f"{API_URL}/predict_illness",
        files={"file": (filename, img_bytes, "image/jpeg")},
        timeout=60,
        retry_delays_seconds=RETRY_DELAYS_SECONDS,
        log_message=f"predict_illness failed | file={filename}",
    )
    return response.json()


def fetch_illness_batch(files: list[dict]) -> dict:
    """Call /predict-set_illness → {filename: {illness, confidence}}."""
    logger.info("predict-set_illness | {} files", len(files))
    response = post_with_retries(
        url=f"{API_URL}/predict-set_illness",
        files=[("files", (f["name"], f["bytes"], "image/jpeg")) for f in files],
        timeout=120,
        retry_delays_seconds=RETRY_DELAYS_SECONDS,
        log_message="predict-set_illness failed",
    )
    results = response.json()
    return {
        item["filename"]: {k: v for k, v in item.items() if k != "filename"}
        for item in results
    }


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
if "ill_predict_batches_loaded" not in st.session_state:
    st.session_state.ill_predict_batches_loaded = set()
if "ill_predict_last_mode" not in st.session_state:
    st.session_state.ill_predict_last_mode = None
if "ill_predict_last_uploader_filenames" not in st.session_state:
    st.session_state.ill_predict_last_uploader_filenames = set()

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.title("Predictions de maladies en lot")
st.markdown("""
    - Cette page vous permet de faire des prédictions sur plusieurs images à la fois.  
    - Il suffit de les sélectionner ci-dessous, puis de naviguer dans les pages de résultats.
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

if st.session_state.ill_predict_last_mode is not None and predict_mode != st.session_state.ill_predict_last_mode:
    reset_batch_page_state(
        session_id=_sid,
        bg_state=_BG_STATE,
        image_files_key="ill_predict_image_files",
        batch_results_key="ill_predict_batch_results",
        batches_loaded_key="ill_predict_batches_loaded",
        page_key="ill_predict_page",
        cache_clear_fn=cached_predict_illness_top3.clear,
    )
st.session_state.ill_predict_last_mode = predict_mode

if predict_mode == MODE_INDIVIDUAL:
    st.info(
        "**Mode individuel** : chaque image est envoyée séparément à l'API. "
        "Vous obtenez les **3 meilleures prédictions** — idéal pour explorer les résultats en détail. "
        "Les résultats s'affichent progressivement au fur et à mesure des appels."
    )
else:
    st.info(
        "**Mode batch** : toutes les images sont envoyées en **une seule requête**. "
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

current_uploader_filenames = {f.name for f in uploaded_images} if uploaded_images else set()
ill_loaded_filenames = {f["name"] for f in st.session_state.ill_predict_image_files}
if current_uploader_filenames and current_uploader_filenames != ill_loaded_filenames:
    st.info(f"**{len(uploaded_images)} image(s) sélectionnée(s)**. Cliquez sur **Load** pour lancer cette sélection.")
elif st.session_state.ill_predict_image_files:
    st.caption(f"{len(st.session_state.ill_predict_image_files)} image(s) chargée(s).")
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
            st.session_state.ill_predict_image_files = image_files
            st.session_state.ill_predict_page = 0
            st.session_state.ill_predict_uploader_key += 1
            st.session_state.ill_predict_batch_results = {}
            st.session_state.ill_predict_batches_loaded = set()
            cached_predict_illness_top3.clear()
            clear_batch_session_tracking(_BG_STATE, _sid)

            if predict_mode == MODE_BATCH:
                first_page = image_files[:PAGE_SIZE]
                first_page_ok = False
                with st.spinner(f"Chargement de la page 1 ({len(first_page)} images)…"):
                    try:
                        results = fetch_illness_batch(first_page)
                        st.session_state.ill_predict_batch_results.update(results)
                        st.session_state.ill_predict_batches_loaded.add(0)
                        st.success(f"Page 1 chargée — {len(results)} images.")
                        first_page_ok = True
                    except Exception as e:
                        st.error(f"Erreur API batch: {e}")
                        logger.error("predict-set_illness error | {}", e)

                remaining_files = image_files[PAGE_SIZE:] if first_page_ok else image_files
                if remaining_files:
                    with _BG_STATE["lock"]:
                        _BG_STATE["running"].add(_sid)
                    threading.Thread(
                        target=run_sequential_subbatch_fetch,
                        args=(_sid, remaining_files, PAGE_SIZE, fetch_illness_batch, _BG_STATE, "bg illness sub-batch fetch failed"),
                        daemon=True,
                    ).start()

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
# Grid — Batch mode: infinite scroll by chunks of PAGE_SIZE
#         Individual mode: paginated
# ---------------------------------------------------------------------------
all_files = st.session_state.ill_predict_image_files
total_files = len(all_files)

_CELL_ILL = "padding:2px 3px; font-size:clamp(0.5rem, 0.95vw, 0.82rem); white-space:nowrap; overflow:hidden; text-overflow:ellipsis; max-width:0"
_HEAD_ILL = f"background:#f5f5f5; {_CELL_ILL}"

if predict_mode == MODE_BATCH:
    # Collect results that are being accumulated in the background
    with _BG_STATE["lock"]:
        st.session_state.ill_predict_batch_results.update(_BG_STATE["results"].get(_sid, {}))
        is_running = _sid in _BG_STATE["running"]
        progress = _BG_STATE["progress"].get(_sid, {"done": 0, "total": 0, "errors": 0})
        failed_files = list(_BG_STATE["failed_files"].get(_sid, []))

    loaded_total = len(st.session_state.ill_predict_batch_results)

    def _render_maladie_item(file: dict, res: dict) -> None:
        pred = res if "illness" in res else next(iter(res.values()))
        low_confidence = pred["confidence"] < min_confidence
        color = confidence_color(pred["confidence"])
        fiche = FICHES_ILL.get(pred["illness"], {})
        display_name = fiche.get("nom_maladie_fr") or pred["illness"]
        conf_icon = "⚠️ " if low_confidence else ""
        st.image(file["bytes"], width="stretch")
        st.caption(file["name"])
        st.markdown(
            f"<div style='text-align:center; width:100%; margin-top:6px; margin-bottom:2px; line-height:1.4'>"
            f"<div style='font-size:clamp(0.85rem, 1.6vw, 1.3rem); font-weight:bold'>{conf_icon}{display_name}</div>"
            f"<div style='font-size:clamp(0.7rem, 1.2vw, 1.0rem); color:{color}; font-weight:bold'>{pred['confidence']:.1%} certitude</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

    render_batch_lot_grids(
        all_files=all_files,
        batch_results=st.session_state.ill_predict_batch_results,
        page_size=PAGE_SIZE,
        grid_cols=GRID_COLS,
        render_item_fn=_render_maladie_item,
    )
    render_batch_progress_footer(
        loaded_total=loaded_total,
        total_files=total_files,
        is_running=is_running,
        progress=progress,
    )

    if not is_running and failed_files:
        st.info(f"{len(failed_files)} image(s) en échec peuvent être relancées sans perdre les résultats déjà reçus.")
        if st.button("Reprendre les lots échoués", use_container_width=True, key="retry_failed_maladie"):
            with _BG_STATE["lock"]:
                _BG_STATE["running"].add(_sid)
            threading.Thread(
                target=run_sequential_subbatch_fetch,
                args=(_sid, failed_files, PAGE_SIZE, fetch_illness_batch, _BG_STATE, "bg illness sub-batch retry failed"),
                daemon=True,
            ).start()
            st.rerun()

    if is_running:
        time.sleep(0.7)
        st.rerun()

else:
    # ── Individual mode: paginated ──────────────────────────────────────
    total_pages = max(1, (total_files + PAGE_SIZE - 1) // PAGE_SIZE)
    page = st.session_state.ill_predict_page
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
                    result = cached_predict_illness_top3(file["bytes"], file["name"])
                except Exception as e:
                    st.image(file["bytes"], width="stretch")
                    st.caption(file["name"])
                    st.error(f"Erreur API: {e}")
                    continue

                top3 = result if isinstance(result, list) else next(iter(result.values()))
                low_confidence = top3[0]["confidence"] < min_confidence
                top1_illness = top3[0]["illness"]
                top1_name = FICHES_ILL.get(top1_illness, {}).get("nom_maladie_fr") or top1_illness
                top1_color = confidence_color(top3[0]["confidence"])
                conf_icon = "⚠️ " if low_confidence else ""
                st.image(file["bytes"], width="stretch")
                st.caption(file["name"])
                st.markdown(
                    f"<div style='text-align:center; width:100%; margin-top:6px; margin-bottom:4px; line-height:1.4'>"
                    f"<div style='font-size:clamp(0.85rem, 1.6vw, 1.3rem); font-weight:bold'>{conf_icon}{top1_name}</div>"
                    f"<div style='font-size:clamp(0.7rem, 1.2vw, 1.0rem); color:{top1_color}; font-weight:bold'>{top3[0]['confidence']:.1%} certitude</div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )
                rows_html = ""
                for rank, pred in enumerate(top3, 1):
                    color = confidence_color(pred["confidence"]) if rank == 1 else "#999"
                    name = FICHES_ILL.get(pred["illness"], {}).get("nom_maladie_fr") or pred["illness"]
                    bg = "background-color:#e8f5e9;" if pred["illness"] == top1_illness else ""
                    rows_html += (
                        f"<tr style='{bg}'>"
                        f"<td style='text-align:center; {_CELL_ILL}'>{rank}</td>"
                        f"<td style='text-align:left; {_CELL_ILL}'>{name}</td>"
                        f"<td style='text-align:right; {_CELL_ILL}; color:{color}; font-weight:bold'>{pred['confidence']:.1%}</td>"
                        f"</tr>"
                    )
                st.markdown(
                    f"<table style='width:100%; table-layout:fixed; border-collapse:collapse; margin-top:4px; margin-bottom:4px; border:1px solid #e0e0e0; border-radius:4px; overflow:hidden'>"
                    f"<colgroup><col style='width:12%'><col style='width:58%'><col style='width:30%'></colgroup>"
                    f"<thead><tr>"
                    f"<th style='text-align:center; {_HEAD_ILL}'>#</th>"
                    f"<th style='text-align:left; {_HEAD_ILL}'>Maladie</th>"
                    f"<th style='text-align:right; {_HEAD_ILL}'>Confiance</th>"
                    f"</tr></thead><tbody>{rows_html}</tbody></table>",
                    unsafe_allow_html=True,
                )
                st.markdown(" ")

    # Pagination (individual mode only)
    st.divider()
    p_left, p_mid, p_right = st.columns([1, 2, 1])
    with p_left:
        if st.button("← Prev", disabled=(page == 0), use_container_width=True):
            st.session_state.ill_predict_page -= 1
            st.rerun()
    with p_mid:
        end_img = min(start + PAGE_SIZE, total_files)
        st.metric("Progression", f"Page {page + 1} / {total_pages}", delta=f"images {start + 1}–{end_img}")
        target_page = st.number_input("Aller à la page", min_value=1, max_value=total_pages,
                                      value=page + 1, step=1, key="ill_predict_jump_page")
        if target_page != page + 1:
            st.session_state.ill_predict_page = int(target_page) - 1
            st.rerun()
    with p_right:
        if st.button("Next →", disabled=(page >= total_pages - 1), use_container_width=True):
            st.session_state.ill_predict_page += 1
            st.rerun()