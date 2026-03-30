import io
import json
import os
import random
from datetime import datetime
from pathlib import Path

import requests
import streamlit as st
from loguru import logger
from PIL import Image

from i18n import get_language, render_language_selector
from styles import COLORS, confidence_color, confidence_badge, styled_info_card, page_header
from utils import post_with_retries, validate_image_file

API_URL = os.environ.get("API_URL", "https://plant-detect-backend-649164185154.europe-west1.run.app")
#API_URL = "http://localhost:8080"
#API_URL = "https://herb-predictor-966041648100.europe-west1.run.app"
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


def _display_illness_name(illness: str) -> str:
    fiche = FICHES.get(illness, {})
    if get_language() == "en":
        return fiche.get("nom_en", illness)
    return fiche.get("nom_maladie_fr", illness)


def _fiche_value(fiche: dict, key: str, language: str) -> str:
    if language == "en":
        return fiche.get(f"{key}_en", fiche.get(key, ""))
    return fiche.get(key, "")


# ---------------------------------------------------------------------------
# Sidebar — prediction history
# ---------------------------------------------------------------------------
with st.sidebar:
    render_language_selector()
    lang = get_language()
    st.markdown("### History" if lang == "en" else "### Historique")
    if not st.session_state.prediction_history:
        st.caption("No predictions yet." if lang == "en" else "Aucune prediction pour le moment.")
    else:
        show_history = st.checkbox("Show history" if lang == "en" else "Afficher l'historique", value=True)
        if st.button("Clear" if lang == "en" else "Effacer", use_container_width=True):
            st.session_state.prediction_history = []
            st.rerun()
        if show_history:
            st.caption(
                f"{len(st.session_state.prediction_history)} / {MAX_HISTORY_ITEMS} items"
                if lang == "en"
                else f"{len(st.session_state.prediction_history)} / {MAX_HISTORY_ITEMS} elements"
            )
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

lang = get_language()
st.title("🌿 Apple and Tomato Disease Predictor" if lang == "en" else "🌿 Predicteur de maladies de pommier ou tomate")
st.markdown(
    "Choose an image from your files or take a photo directly with your camera. "
    "The plant disease recognition model returns a real-time prediction with confidence."
    if lang == "en"
    else "Vous pouvez soit choisir une image dans vos dossiers, soit prendre une photo directement avec votre camera. "
    "Le modele de reconnaissance de maladie de plantes vous donnera une prediction en temps reel avec un score de confiance."
)
st.markdown(
    "To predict multiple images at once, go to the 'Batch Predict' page."
    if lang == "en"
    else "Pour predire plusieurs images a la fois, rendez-vous dans l'onglet 'Batch Predict'."
)

# ---------------------------------------------------------------------------
# Input
# ---------------------------------------------------------------------------
tab_upload, tab_camera = st.tabs(["📁 Upload image", "📷 Camera"])

uploaded_file = None

with tab_upload:
    f = st.file_uploader(
        "Choose an image (jpg/jpeg/png)" if lang == "en" else "Choisissez une image (jpg/jpeg/png)",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=False,
        label_visibility="collapsed",
    )
    if f:
        uploaded_file = f

with tab_camera:
    photo = st.camera_input("Take a photo" if lang == "en" else "Prenez une photo", label_visibility="collapsed")
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
st.markdown(
    "##### When you are ready, click the button below to launch prediction. This may take up to 1 minute depending on server load and image complexity."
    if lang == "en"
    else "##### Quand vous serez pret, vous pouvez cliquer sur le bouton ci-dessous pour lancer la prediction. "
    "En fonction de la charge du serveur et de la complexite de l'image, cela peut prendre jusqu'a 1 minute. Merci pour votre patience !"
)

if st.button("🔍 Identify", type="primary", use_container_width=False):
    with st.spinner("Analyzing (~30-60s)..." if lang == "en" else "Analyse en cours (~30-60s)..."):
        try:
            logger.info("predict_illness | file={}", uploaded_file.name)
            file_bytes = uploaded_file.getvalue()
            response = post_with_retries(
                url=f"{API_URL}/predict_illness",
                files={"file": (uploaded_file.name, file_bytes, uploaded_file.type or "image/jpeg")},
                timeout=60,
                retry_delays_seconds=RETRY_DELAYS_SECONDS,
                log_message="predict_illness failed",
            )
        except requests.exceptions.ConnectionError:
            st.error("Unable to reach the API. Please verify the service is online." if lang == "en" else "Impossible de joindre l'API. Verifiez que le service est en ligne.")
            logger.error("API connection error | url={}", API_URL)
            st.stop()
        except requests.exceptions.Timeout:
            st.error("The service is taking too long to respond. Please retry in a few seconds." if lang == "en" else "Le service met trop de temps a repondre. Reessayez dans quelques secondes.")
            st.stop()
        except requests.exceptions.HTTPError as e:
            st.error((f"API error: {e}") if lang == "en" else (f"Erreur API: {e}"))
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
    st.subheader("Results" if lang == "en" else "Resultats")
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
            st.markdown(f"### About — {nom_en_md} (*{nom_fr_md}*)" if lang == "en" else f"### A propos - {nom_fr_md} (*{nom_en_md}*)")
            

            info_dict = {
                ("🦠 Possible cause" if lang == "en" else "🦠 Cause possible"): _fiche_value(fiche, "cause", lang),
                ("🩺 Curative treatment" if lang == "en" else "🩺 Traitement curatif"): _fiche_value(fiche, "traitement_curatif", lang),
                ("💊 Preventive treatment" if lang == "en" else "💊 Traitement preventif"): _fiche_value(fiche, "traitement_preventif", lang),
                ("🛡️ Season / Severity" if lang == "en" else "🛡️ Saison / Gravite"): _fiche_value(fiche, "saison_gravite", lang),
            }
            styled_info_card("More information" if lang == "en" else "Plus d'informations", info_dict)
        else:
            st.markdown(f"### {top_illness}")
            st.markdown("No profile is available for this disease." if lang == "en" else "Aucune fiche disponible pour cette maladie.")
            wiki_search = f"https://fr.wikipedia.org/wiki/Special:Search?search={top_illness.replace(' ', '+')}"
            st.markdown(f"[Search on Wikipedia]({wiki_search})" if lang == "en" else f"[Rechercher sur Wikipedia]({wiki_search})")

        
        if top_confidence < 0.50:
            st.info("Low confidence: try a sharper photo, better lighting, and a closer crop on the leaf." if lang == "en" else "Confiance faible: essayez une photo plus nette, une meilleure lumiere et un cadrage plus serre sur la plante.")
            
    # ── more information on the predictions ─────────────────────────────────────────────────────
    st.text(" ")  # Spacer
    st.divider()
    st.markdown("### Prediction Details" if lang == "en" else "### Details des predictions")
    
    with st.expander("View prediction details" if lang == "en" else "Voir les details des predictions"):
        models_list = list(models_used)
        for i in range(0, len(models_list), 2):
            grid_cols = st.columns(2)
            for j, key in enumerate(models_list[i:i+2]):
                with grid_cols[j]:
                    st.markdown(f"#### **Model: {key.upper()}**" if lang == "en" else f"#### **Modele: {key.upper()}**")
                    species    = data[key][0]["illness"]
                    confidence = data[key][0]["confidence"]
                    color = confidence_color(confidence)

                    st.markdown(
                        f"<p style='font-size:1.4rem; font-weight:700; margin:0'>{_display_illness_name(species)}</p>"
                        f"<p style='color:{color}; font-size:1.1rem; margin:4px 0 16px'>"
                        f"{confidence:.0%} confidence</p>",
                        unsafe_allow_html=True,
                    )

                    st.markdown("**Top 3**")
                    for rank, pred in enumerate(data[key], 1):
                        bar_pct = int(pred["confidence"] * 100)
                        st.markdown(f"**{rank}. {_display_illness_name(pred['illness'])}** — {pred['confidence']:.0%}")
                        st.progress(bar_pct)