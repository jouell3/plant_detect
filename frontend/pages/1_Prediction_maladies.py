import io
import json
import os
import random
import time
from datetime import datetime
from pathlib import Path

import requests
import streamlit as st
from loguru import logger
from PIL import Image

from styles import COLORS, confidence_color, confidence_badge, styled_info_card, page_header
from utils import validate_image_file

#API_URL = os.environ.get("API_URL", "https://plant-detect-backend-649164185154.europe-west1.run.app")
#API_URL = "http://localhost:8080"
API_URL = "https://herb-predictor-966041648100.europe-west1.run.app"
MAX_HISTORY_ITEMS = 20
RETRY_DELAYS_SECONDS = (0.8, 1.6)


_FICHES_PATH = Path(__file__).parent.parent / "fiches_ill.json"
FICHES: dict = json.loads(_FICHES_PATH.read_text(encoding="utf-8")) if _FICHES_PATH.exists() else {}

st.set_page_config(page_title="Maladie Predictor", layout="wide")

# ---------------------------------------------------------------------------
# Session state — prediction history
# ---------------------------------------------------------------------------
if "prediction_history" not in st.session_state:
    st.session_state.prediction_history = []  # list of {name, species, confidence, thumb_bytes, timestamp}
if "last_prediction" not in st.session_state:
    st.session_state.last_prediction = None
if "suggestions_pool" not in st.session_state:
    st.session_state.suggestions_pool = []
if "suggestions_visible_count" not in st.session_state:
    st.session_state.suggestions_visible_count = 0
if "suggestions_species_key" not in st.session_state:
    st.session_state.suggestions_species_key = ""
if "last_uploaded_id" not in st.session_state:
    st.session_state.last_uploaded_id = None


def _normalize_species_key(value: str) -> str:
    return (value or "").strip().lower().replace("-", " ")


def _post_predict_with_retries(file_name: str, file_bytes: bytes, file_mime: str | None):
    last_error = None
    for idx, delay in enumerate((0.0, *RETRY_DELAYS_SECONDS)):
        if delay > 0:
            time.sleep(delay)
        try:
            response = requests.post(
                f"{API_URL}/predict_illness",
                files={"file": (file_name, file_bytes, file_mime or "image/jpeg")},
                timeout=60,
            )
            response.raise_for_status()
            return response
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout, requests.exceptions.HTTPError) as e:
            last_error = e
            logger.warning("predict_illness failed | attempt={} | error={}", idx + 1, e)
    raise last_error

# ---------------------------------------------------------------------------
# Sidebar — prediction history
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### Historique")
    if not st.session_state.prediction_history:
        st.caption("Aucune prédiction pour le moment.")
    else:
        show_history = st.checkbox("Afficher l'historique", value=True)
        if st.button("Effacer", use_container_width=True):
            st.session_state.prediction_history = []
            st.rerun()
        if show_history:
            st.caption(f"{len(st.session_state.prediction_history)} / {MAX_HISTORY_ITEMS} éléments")
            for entry in reversed(st.session_state.prediction_history):
                conf = entry["confidence"]
                color = confidence_color(conf)
                st.image(entry["thumb_bytes"], width=60)
                st.markdown(
                    f"**{entry['species']}**  \n"
                    f"<span style='color:{color}'>{conf:.0%}</span> · "
                    f"<span style='font-size:0.75rem; color:#616161'>{entry['timestamp']}</span>",
                    unsafe_allow_html=True,
                )
                st.divider()

st.title("🌿 Maladie Predictor")
st.markdown("Vous pouvez soit choisir une image dans vos dossiers, soit prendre une photo directement avec votre caméra. Le modèle de reconnaissance de maladie de plantes vous donnera une prédiction en temps réel avec un score de confiance. N'hésitez pas à tester plusieurs images pour voir les résultats !")
st.markdown("Pour predire plusieurs images à la fois, rendez-vous dans l'onglet 'Batch Predict'.")

# ---------------------------------------------------------------------------
# Input
# ---------------------------------------------------------------------------
tab_upload, tab_camera = st.tabs(["📁 Upload image", "📷 Camera"])

uploaded_file = None

with tab_upload:
    f = st.file_uploader(
        "Choisissez une image (jpg/jpeg/png)",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=False,
        label_visibility="collapsed",
    )
    if f:
        uploaded_file = f

with tab_camera:
    photo = st.camera_input("Prenez une photo", label_visibility="collapsed")
    if photo:
        uploaded_file = photo

# ---------------------------------------------------------------------------
# Predict
# ---------------------------------------------------------------------------
if not uploaded_file:
    st.stop()

# Clear previous prediction when a new file is selected
_file_id = (uploaded_file.name, uploaded_file.size)
if _file_id != st.session_state.last_uploaded_id:
    st.session_state.last_prediction = None
    st.session_state.last_uploaded_id = _file_id

# Validate uploaded image before proceeding
is_valid, error_msg = validate_image_file(uploaded_file)
if not is_valid:
    st.error(f"❌ {error_msg}")
    st.stop()

st.divider()
st.markdown("##### Quand vous serez prêt, vous pouvez cliquer sur le bouton ci-dessous pour lancer la prédiction. En fonction de la charge du serveur et de la complexité de l'image, cela peut prendre jusqu'à 1 minute. Merci pour votre patience !")

if st.button("🔍 Identify", type="primary", use_container_width=False):
    with st.spinner("Analyse en cours (~30-60s)…"):
        try:
            logger.info("predict_illness | file={}", uploaded_file.name)
            file_bytes = uploaded_file.getvalue()
            response = _post_predict_with_retries(uploaded_file.name, file_bytes, uploaded_file.type)
        except requests.exceptions.ConnectionError:
            st.error("Impossible de joindre l'API. Vérifiez que le service est en ligne.")
            logger.error("API connection error | url={}", API_URL)
            st.stop()
        except requests.exceptions.Timeout:
            st.error("Le service met trop de temps à répondre. Réessayez dans quelques secondes.")
            st.stop()
        except requests.exceptions.HTTPError as e:
            st.error(f"Erreur API: {e}")
            logger.error("API HTTP error | {}", e)
            st.stop()

    data = response.json()  # {model: [{species, confidence}, ...]}
    models_used = list(data.keys())
    first_model = models_used[0]
    top_illness = data[first_model][0]["illness"]
    top_confidence = data[first_model][0]["confidence"]

    st.session_state.last_prediction = {
        "data": data,
        "models_used": models_used,
        "top_illness": top_illness,
        "top_confidence": top_confidence,
        "uploaded_name": uploaded_file.name,
        "uploaded_bytes": file_bytes,
    }

    species_key = _normalize_species_key(top_illness)

    # ── Record to history ─────────────────────────────────────────────────
    st.session_state.prediction_history.append({
        "name": uploaded_file.name,
        "species": top_illness,
        "confidence": top_confidence,
        "thumb_bytes": uploaded_file.getvalue(),
        "timestamp": datetime.now().strftime("%H:%M:%S"),
    })
    if len(st.session_state.prediction_history) > MAX_HISTORY_ITEMS:
        st.session_state.prediction_history = st.session_state.prediction_history[-MAX_HISTORY_ITEMS:]

prediction = st.session_state.last_prediction
if prediction:
    data = prediction["data"]
    models_used = prediction["models_used"]
    top_illness = prediction["top_illness"]
    top_confidence = prediction["top_confidence"]

    # ── Display ───────────────────────────────────────────────────────────
    st.subheader("Résultats")
    _, col_img, col_fiche, _ = st.columns([0.2, 1, 3, 0.2], gap="large", vertical_alignment="bottom")

    with col_img:
        img = Image.open(io.BytesIO(prediction["uploaded_bytes"]))
        st.image(img, caption=prediction["uploaded_name"], width="stretch")

    with col_fiche:
    # ── Herb info card ─────────────────────────────────────────────────────
        illness_found = [data[key][0]["illness"] for key in models_used]
        
        fiche = FICHES.get(top_illness)
        #fiche = FICHES.get(_normalize_species_key(top_illness))
        st.divider()
        if fiche:
            nom_fr_md = f"[{fiche['nom_maladie_fr']}]({fiche['wikipedia_fr']})" if fiche.get("wikipedia_fr") else fiche['nom_maladie_fr']
            nom_en_md = f"[{fiche['nom_en']}]({fiche['wikipedia_en']})" if fiche.get("wikipedia_en") else fiche['nom_en']
            st.markdown(f"### À propos — {nom_fr_md} (*{nom_en_md}*)")
            

            info_dict = {
                "🦠 Cause possible": fiche['cause'],
                "🩺 Traitement curatif": fiche['traitement_curatif'],
                "💊 Traitement préventif": fiche['traitement_preventif'],
                "🛡️ Saison / Gravité": fiche['saison_gravite'],
            }
            styled_info_card("Plus d'informations", info_dict)
        else:
            st.markdown(f"### {top_illness}")
            st.markdown("Aucune fiche disponible pour cette maladie.")
            wiki_search = f"https://fr.wikipedia.org/wiki/Special:Search?search={top_illness.replace(' ', '+')}"
            st.markdown(f"[Rechercher sur Wikipedia]({wiki_search})")

        
        if top_confidence < 0.50:
            st.info("Confiance faible: essayez une photo plus nette, une meilleure lumière et un cadrage plus serré sur la plante.")
            
    # ── more information on the predictions ─────────────────────────────────────────────────────
    st.text(" ")  # Spacer
    st.divider()
    st.markdown("### Détails des prédictions")
    
    with st.expander("Voir les détails des prédictions"):
        models_list = list(models_used)
        for i in range(0, len(models_list), 2):
            grid_cols = st.columns(2)
            for j, key in enumerate(models_list[i:i+2]):
                with grid_cols[j]:
                    st.markdown(f"#### **Modèle: {key.upper()}**")
                    species    = data[key][0]["illness"]
                    confidence = data[key][0]["confidence"]
                    color = confidence_color(confidence)

                    st.markdown(
                        f"<p style='font-size:1.4rem; font-weight:700; margin:0'>{species}</p>"
                        f"<p style='color:{color}; font-size:1.1rem; margin:4px 0 16px'>"
                        f"{confidence:.0%} confidence</p>",
                        unsafe_allow_html=True,
                    )

                    st.markdown("**Top 3**")
                    for rank, pred in enumerate(data[key], 1):
                        bar_pct = int(pred["confidence"] * 100)
                        st.markdown(f"**{rank}. {pred['illness']}** — {pred['confidence']:.0%}")
                        st.progress(bar_pct)