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

# Local imports for styling and validation
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from styles import COLORS, confidence_color, confidence_badge, styled_info_card, page_header
from utils import validate_image_file
from herbs_detection.model_illness import predict_top3 as ill_top3
from herbs_detection.model_illness import load_model as ill_load_model

ill_load_model()  # Load the illness prediction model at startup to reduce latency on first prediction

#
# API_URL = os.environ.get("API_URL", "https://plant-detect-backend-649164185154.europe-west1.run.app")
MAX_HISTORY_ITEMS = 20
RETRY_DELAYS_SECONDS = (0.8, 1.6)


_FICHES_PATH = Path(__file__).parent.parent / "fiches.json"
FICHES: dict = json.loads(_FICHES_PATH.read_text(encoding="utf-8")) if _FICHES_PATH.exists() else {}
_SUGGESTIONS_PATH = Path(__file__).parent.parent / "suggestions.json"
SUGGESTIONS: dict = json.loads(_SUGGESTIONS_PATH.read_text(encoding="utf-8")) if _SUGGESTIONS_PATH.exists() else {}

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


def _normalize_species_key(value: str) -> str:
    return (value or "").strip().lower().replace("-", " ")


def _post_predict_with_retries(file_name: str, file_bytes: bytes, file_mime: str | None):
    last_error = None
    for idx, delay in enumerate((0.0, *RETRY_DELAYS_SECONDS)):
        if delay > 0:
            time.sleep(delay)
        try:
            response = requests.post(
                f"{API_URL}/predict_herb",
                files={"file": (file_name, file_bytes, file_mime or "image/jpeg")},
                timeout=60,
            )
            response.raise_for_status()
            return response
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout, requests.exceptions.HTTPError) as e:
            last_error = e
            logger.warning("predict_herb failed | attempt={} | error={}", idx + 1, e)
    raise last_error


def _generate_recipe_prompt(dish_name: str, herb_name: str) -> str:
    """Generate a detailed recipe prompt for an AI chat."""
    return f"Donne-moi une recette complète et détaillée pour {dish_name} en utilisant du {herb_name.lower()} frais. Inclus les ingrédients, les instructions étape par étape, et les conseils de cuisson."


def _get_suggestions_for_species(species_key: str) -> list[dict]:
    entry = SUGGESTIONS.get(species_key, [])
    if isinstance(entry, list):
        return entry
    if isinstance(entry, dict):
        suggestions = entry.get("suggestions", [])
        return suggestions if isinstance(suggestions, list) else []
    return []

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

# Validate uploaded image before proceeding
is_valid, error_msg = validate_image_file(uploaded_file)
if not is_valid:
    st.error(f"❌ {error_msg}")
    st.stop()

st.divider()
st.markdown("##### Quand vous serez prêt, vous pouvez cliquer sur le bouton ci-dessous pour lancer la prédiction. En fonction de la charge du serveur et de la complexité de l'image, cela peut prendre entre 30 secondes et 1 minute. Merci pour votre patience !")

if st.button("🔍 Identify", type="primary", use_container_width=False):
    with st.spinner("Analyse en cours (~30-60s)…"):
        try:
            logger.info("predict_herb | file={}", uploaded_file.name)
            file_bytes = uploaded_file.getvalue()
            reponse = ill_top3(uploaded_file) # Call local prediction function with retries
            st.markdown(reponse)
            #response = _post_predict_with_retries(uploaded_file.name, file_bytes, uploaded_file.type)
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

    #data = response.json()  # {model: [{species, confidence}, ...]}
    models_used = "Pytorch ResNet18"
    #models_used = list(data.keys())
    #first_model = models_used[0]
    #top_illness = data[first_model][0]["species"]
    #top_confidence = data[first_model][0]["confidence"]
    top_illness = reponse[0][0]
    top_confidence = reponse[0][1]

    st.session_state.last_prediction = {
        #"data": data,
        "models_used": models_used,
        "top_illness": top_illness,
        "top_confidence": top_confidence,
        "uploaded_name": uploaded_file.name,
        "uploaded_bytes": file_bytes,
    }

    species_key = _normalize_species_key(top_illness)
    #all_suggestions = _get_suggestions_for_species(species_key)
    #shuffled_suggestions = list(all_suggestions)
    #random.shuffle(shuffled_suggestions)
    #st.session_state.suggestions_pool = shuffled_suggestions
    #st.session_state.suggestions_visible_count = min(6, len(shuffled_suggestions), 12)
    #st.session_state.suggestions_species_key = species_key

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
    col_img, col_fiche = st.columns([1, 3])

    with col_img:
        img = Image.open(io.BytesIO(prediction["uploaded_bytes"]))
        st.image(img, width=450, caption=prediction["uploaded_name"])

    with col_fiche:
    # ── Herb info card ─────────────────────────────────────────────────────
        herb_found = [data[key][0]["species"] for key in models_used]
        
        if len(set(herb_found)) == 1:  # If both models agree on the same herb, show the info card
            fiche = FICHES.get(_normalize_species_key(top_illness))
            st.divider()
            if fiche:
                nom_fr_md = f"[{fiche['nom_fr']}]({fiche['wikipedia_fr']})" if fiche.get("wikipedia_fr") else fiche['nom_fr']
                nom_en_md = f"[{fiche['nom_en']}]({fiche['wikipedia_en']})" if fiche.get("wikipedia_en") else fiche['nom_en']
                st.markdown(f"### À propos — {nom_fr_md} (*{nom_en_md}*)")
                st.markdown(fiche["description"])

                info_dict = {
                    "🌸 Arôme": fiche['arome'],
                    "🌱 Culture": fiche['culture'],
                    "⚠️ Toxicité": fiche['toxicite'],
                    "🍽️ Usages": ', '.join(fiche['usages']),
                    "🤝 Compatible": ', '.join(fiche['compatibilites']),
                }
                styled_info_card("Propriétés", info_dict)
            else:
                st.markdown(f"### {top_illness}")
                st.markdown("Aucune fiche disponible pour cette maladie.")
                wiki_search = f"https://fr.wikipedia.org/wiki/Special:Search?search={top_illness.replace(' ', '+')}"
                st.markdown(f"[Rechercher sur Wikipedia]({wiki_search})")

        if len(set(herb_found)) != 1:
            st.warning(f"Les {len(list(models_used))} modèles ne sont pas d'accord sur la prédiction. Veuillez essayer avec une autre image ou prendre la photo dans de meilleures conditions d'éclairage ou d'angle.")
        elif top_confidence < 0.50:
            st.info("Confiance faible: essayez une photo plus nette, une meilleure lumière et un cadrage plus serré sur la plante.")
    
    # ── Suggestions grid section ──────────────────────────────────────────────
    if len(set(herb_found)) == 1:
        species_key = _normalize_species_key(top_illness)
        if st.session_state.suggestions_species_key == species_key and st.session_state.suggestions_pool:
            max_display = min(12, len(st.session_state.suggestions_pool))
            visible_count = min(st.session_state.suggestions_visible_count, max_display)
            suggestions = st.session_state.suggestions_pool[:visible_count]
            st.divider()
            st.markdown("### 💡 Suggestions d'utilisation")
            st.markdown(f"Voici des idées de plats à préparer avec du **{fiche['nom_fr']}** frais. Cliquez sur les boutons pour générer des prompts de recettes détaillées à utiliser avec votre chat IA préféré (ChatGPT, Claude, etc.).")
            # Create 3-column grid layout
            for idx in range(0, len(suggestions), 3):
                cols = st.columns(3)
                for col_idx, suggestion in enumerate(suggestions[idx:idx+3]):
                    with cols[col_idx]:
                        suggestion_html = f"<div style='background: #f9f9f9; border-left: 4px solid {COLORS['success']}; padding: 16px; margin: 0; border-radius: 4px;'><div style='font-size: 1.1rem; font-weight: 700; color: {COLORS['text_primary']}; margin-bottom: 8px;'>{suggestion['plat']}</div><div style='font-size: 0.9rem; color: {COLORS['text_muted']}; line-height: 1.6;'>{suggestion['description']}</div></div>"
                        st.markdown(suggestion_html, unsafe_allow_html=True)
                        
                        # Generate recipe prompt in expandable section
                        prompt = _generate_recipe_prompt(suggestion['plat'], fiche['nom_fr'])
                        with st.expander(f"📋 Pour générer la recette de {suggestion['plat']}, clicker ici", expanded=False):
                            st.code(prompt, language="text")
                            st.caption("Copiez ce texte et collez-le dans votre chat IA préféré (ChatGPT, Claude, etc.)")

            if st.button("Suggère 3 de plus", disabled=visible_count >= max_display):
                st.session_state.suggestions_visible_count = min(visible_count + 3, max_display)
                st.rerun()
        
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
                    species    = data[key][0]["species"]
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
                        st.markdown(f"**{rank}. {pred['species']}** — {pred['confidence']:.0%}")
                        st.progress(bar_pct)