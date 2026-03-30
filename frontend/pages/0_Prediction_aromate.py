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


_FICHES_PATH = Path(__file__).parent.parent / "fiches.json"
FICHES: dict = json.loads(_FICHES_PATH.read_text(encoding="utf-8")) if _FICHES_PATH.exists() else {}
_SUGGESTIONS_PATH = Path(__file__).parent.parent / "suggestions.json"
SUGGESTIONS: dict = json.loads(_SUGGESTIONS_PATH.read_text(encoding="utf-8")) if _SUGGESTIONS_PATH.exists() else {}

st.set_page_config(page_title="Plant Predictor", layout="wide")

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


def _display_species_name(species: str) -> str:
    fiche = FICHES.get(_normalize_species_key(species), {})
    if get_language() == "en":
        return fiche.get("nom_en", fiche.get("nom_fr", species))
    return fiche.get("nom_fr", species)


def _fiche_value(fiche: dict, key: str, language: str):
    if language == "en":
        return fiche.get(f"{key}_en", fiche.get(key))
    return fiche.get(key)


def _suggestion_value(suggestion: dict, key: str, language: str) -> str:
    if language == "en":
        return suggestion.get(f"{key}_en", suggestion.get(key, ""))
    return suggestion.get(key, "")


def _generate_recipe_prompt(dish_name: str, herb_name: str) -> str:
    """Generate a detailed recipe prompt for an AI chat."""
    if get_language() == "en":
        return (
            f"Give me a complete and detailed recipe for {dish_name} using fresh {herb_name.lower()}. "
            "Include ingredients, step-by-step instructions, and cooking tips."
        )
    return (
        f"Donne-moi une recette complete et detaillee pour {dish_name} en utilisant du {herb_name.lower()} frais. "
        "Inclus les ingredients, les instructions etape par etape, et les conseils de cuisson."
    )


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
st.title("🌿 Aromatic Herb Predictor" if lang == "en" else "🌿 Predicteur d'aromate")
st.markdown(
    "Choose an image from your files or take a photo directly with your camera. "
    "The aromatic herb recognition model will return a real-time prediction with a confidence score. "
    "Feel free to test several images."
    if lang == "en"
    else "Vous pouvez soit choisir une image dans vos dossiers, soit prendre une photo directement avec votre camera. "
    "Le modele de reconnaissance d'herbes aromatiques vous donnera une prediction en temps reel avec un score de confiance. "
    "N'hesitez pas a tester plusieurs images pour voir les resultats !"
)
st.markdown(
    "To predict multiple images at once, go to the 'Batch Predict' page."
    if lang == "en"
    else "Pour predire plusieurs images a la fois, rendez-vous dans l'onglet 'Batch Predict'."
)

# ---------------------------------------------------------------------------
# Input
# ---------------------------------------------------------------------------
tab_upload, tab_camera = st.tabs([
    "📁 Upload image" if lang == "en" else "📁 Upload image",
    "📷 Camera" if lang == "en" else "📷 Camera",
])

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
    "##### When you are ready, click the button below to launch prediction. "
    "Depending on server load and image complexity, this can take 30 to 60 seconds."
    if lang == "en"
    else "##### Quand vous serez pret, vous pouvez cliquer sur le bouton ci-dessous pour lancer la prediction. "
    "En fonction de la charge du serveur et de la complexite de l'image, cela peut prendre entre 30 secondes et 1 minute. "
    "Merci pour votre patience !"
)


if st.button("🔍 Identify", type="primary", use_container_width=False):
    with st.spinner("Analyzing (~30-60s)..." if lang == "en" else "Analyse en cours (~30-60s)..."):
        try:
            logger.info("predict_herb | file={}", uploaded_file.name)
            file_bytes = uploaded_file.getvalue()
            response = post_with_retries(
                url=f"{API_URL}/predict_herb",
                files={"file": (uploaded_file.name, file_bytes, uploaded_file.type or "image/jpeg")},
                timeout=60,
                retry_delays_seconds=RETRY_DELAYS_SECONDS,
                log_message="predict_herb failed",
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
    top_species = []
    mean_confidence = []
    good_models = []
    for key in models_used:
        top_species.append(data[key][0]["species"])
    top_species = max(set(top_species), key=top_species.count)  # Most common top species among models
    other_species = [s for s in top_species if s != top_species]
    for key in models_used:
        if data[key][0]["species"] == top_species:
            mean_confidence.append(data[key][0]["confidence"])
            good_models.append(key)
    mean_confidence = sum(mean_confidence) / len(mean_confidence) if mean_confidence else 0.0

    st.session_state.last_prediction = {
        "data": data,
        "models_used": models_used,
        "top_species": top_species,
        "mean_confidence": mean_confidence,
        "good_models": good_models,
        "uploaded_name": uploaded_file.name,
        "uploaded_bytes": file_bytes,
    }

    species_key = _normalize_species_key(top_species)
    all_suggestions = _get_suggestions_for_species(species_key)
    shuffled_suggestions = list(all_suggestions)
    random.shuffle(shuffled_suggestions)
    st.session_state.suggestions_pool = shuffled_suggestions
    st.session_state.suggestions_visible_count = min(6, len(shuffled_suggestions), 12)
    st.session_state.suggestions_species_key = species_key

    # ── Record to history ─────────────────────────────────────────────────
    st.session_state.prediction_history.append({
        "name": uploaded_file.name,
        "species": top_species,
        "confidence": mean_confidence,
        "thumb_bytes": uploaded_file.getvalue(),
        "timestamp": datetime.now().strftime("%H:%M:%S"),
    })
    if len(st.session_state.prediction_history) > MAX_HISTORY_ITEMS:
        st.session_state.prediction_history = st.session_state.prediction_history[-MAX_HISTORY_ITEMS:]

prediction = st.session_state.last_prediction
if prediction:
    data = prediction["data"]
    models_used = prediction["models_used"]
    top_species = prediction["top_species"]
    mean_confidence = prediction["mean_confidence"]
    good_models = prediction["good_models"]
    top_species_fr = FICHES.get(_normalize_species_key(top_species), {}).get("nom_fr", top_species)

    # ── Display ───────────────────────────────────────────────────────────
    st.subheader("Results" if lang == "en" else "Resultats")
    herb_found = [data[key][0]["species"] for key in models_used]
    
    if len(set(herb_found)) == 2:
        st.warning(
            "At least one model disagrees with the prediction, but here is the result supported by the other models. "
            "You can review each model's details below. Try another image or improve lighting and angle."
            if lang == "en"
            else "Au moins un des modeles n'est pas d'accord sur la prediction mais voici quand meme la prediction obtenue pour les autres modeles. "
            "Vous pouvez voir les details de chaque prediction en bas de page. Essayez avec une autre image ou prenez la photo dans de meilleures conditions d'eclairage ou d'angle."
        )
    if len(set(herb_found)) <= 2:
        st.success(
            f"The model predicted **{_display_species_name(top_species).lower()}** with an average confidence of {mean_confidence:.0%} across {len(good_models)} models: {', '.join(good_models)}."
            if lang == "en"
            else f"Le modele a predit **{top_species_fr.lower()}** avec une confiance moyenne de {mean_confidence:.0%} sur les {len(good_models)} modeles suivants: {', '.join(good_models)}."
        )
    
    col_img, col_fiche= st.columns([ 1, 3], vertical_alignment="bottom", gap="large")

    with col_img:
        img = Image.open(io.BytesIO(prediction["uploaded_bytes"]))
        st.image(img, width="stretch", caption=prediction["uploaded_name"])

    with col_fiche:
    # ── Herb info card ─────────────────────────────────────────────────────
        #herb_found = [data[key][0]["species"] for key in models_used]
        
        if len(set(herb_found)) <= 2:  # If at least 3 models out of the 4 agree on the same herb, show the info card
            fiche = FICHES.get(_normalize_species_key(top_species))
            st.divider()
            if fiche:
                nom_fr_md = f"[{fiche['nom_fr']}]({fiche['wikipedia_fr']})" if fiche.get("wikipedia_fr") else fiche['nom_fr']
                nom_en_md = f"[{fiche['nom_en']}]({fiche['wikipedia_en']})" if fiche.get("wikipedia_en") else fiche['nom_en']
                st.markdown(f"### About — {nom_en_md} (*{nom_fr_md}*)" if lang == "en" else f"### A propos - {nom_fr_md} (*{nom_en_md}*)")
                st.markdown(_fiche_value(fiche, "description", lang))

                info_dict = {
                    ("🌸 Aroma" if lang == "en" else "🌸 Arome"): _fiche_value(fiche, "arome", lang),
                    ("🌱 Cultivation" if lang == "en" else "🌱 Culture"): _fiche_value(fiche, "culture", lang),
                    ("⚠️ Toxicity" if lang == "en" else "⚠️ Toxicite"): _fiche_value(fiche, "toxicite", lang),
                    ("🍽️ Uses" if lang == "en" else "🍽️ Usages"): ', '.join(_fiche_value(fiche, "usages", lang)),
                    ("🤝 Pairings" if lang == "en" else "🤝 Compatible"): ', '.join(_fiche_value(fiche, "compatibilites", lang)),
                }
                styled_info_card("Properties" if lang == "en" else "Proprietes", info_dict)
            else:
                st.markdown(f"### {top_species}")
                st.markdown("No profile is available for this plant." if lang == "en" else "Aucune fiche disponible pour cette plante.")
                wiki_search = f"https://fr.wikipedia.org/wiki/Special:Search?search={top_species.replace(' ', '+')}"
                st.markdown(f"[Search on Wikipedia]({wiki_search})" if lang == "en" else f"[Rechercher sur Wikipedia]({wiki_search})")

        if len(set(herb_found)) > 2:
            st.warning(
                f"The {len(list(models_used))} models disagree on the prediction. Please try another image or improve lighting and angle."
                if lang == "en"
                else f"Les {len(list(models_used))} modeles ne sont pas d'accord sur la prediction. Veuillez essayer avec une autre image ou prendre la photo dans de meilleures conditions d'eclairage ou d'angle."
            )
        elif mean_confidence < 0.50:
            st.info("Low confidence: try a sharper photo, better lighting, and a closer crop on the plant." if lang == "en" else "Confiance faible: essayez une photo plus nette, une meilleure lumiere et un cadrage plus serre sur la plante.")
    
    # ── Suggestions grid section ──────────────────────────────────────────────
    if len(set(herb_found)) <= 2:
        species_key = _normalize_species_key(top_species)
        if st.session_state.suggestions_species_key == species_key and st.session_state.suggestions_pool:
            max_display = min(12, len(st.session_state.suggestions_pool))
            visible_count = min(st.session_state.suggestions_visible_count, max_display)
            suggestions = st.session_state.suggestions_pool[:visible_count]
            st.divider()
            st.markdown("### 💡 Usage Suggestions" if lang == "en" else "### 💡 Suggestions d'utilisation")
            herb_display_name = fiche['nom_en'] if lang == "en" else fiche['nom_fr']
            st.markdown(
                f"Here are dish ideas to prepare with fresh **{herb_display_name}**. Click the sections below to generate detailed recipe prompts for your preferred AI chat (ChatGPT, Claude, etc.)."
                if lang == "en"
                else f"Voici des idees de plats a preparer avec du **{herb_display_name}** frais. Cliquez sur les boutons pour generer des prompts de recettes detaillees a utiliser avec votre chat IA prefere (ChatGPT, Claude, etc.)."
            )
            # Create 3-column grid layout
            for idx in range(0, len(suggestions), 3):
                cols = st.columns(3)
                for col_idx, suggestion in enumerate(suggestions[idx:idx+3]):
                    with cols[col_idx]:
                        dish_name = _suggestion_value(suggestion, "plat", lang)
                        dish_description = _suggestion_value(suggestion, "description", lang)
                        suggestion_html = f"<div style='background: #f9f9f9; border-left: 4px solid {COLORS['success']}; padding: 16px; margin: 0; border-radius: 4px;'><div style='font-size: 1.1rem; font-weight: 700; color: {COLORS['text_primary']}; margin-bottom: 8px;'>{dish_name}</div><div style='font-size: 0.9rem; color: {COLORS['text_muted']}; line-height: 1.6;'>{dish_description}</div></div>"
                        st.markdown(suggestion_html, unsafe_allow_html=True)
                        
                        # Generate recipe prompt in expandable section
                        prompt = _generate_recipe_prompt(dish_name, herb_display_name)
                        with st.expander(
                            f"📋 Click to generate the recipe for {dish_name}" if lang == "en" else f"📋 Pour generer la recette de {dish_name}, cliquez ici",
                            expanded=False,
                        ):
                            st.code(prompt, language="text")
                            st.caption("Copy this text and paste it into your preferred AI chat (ChatGPT, Claude, etc.)" if lang == "en" else "Copiez ce texte et collez-le dans votre chat IA prefere (ChatGPT, Claude, etc.)")

            if st.button("Suggest 3 more" if lang == "en" else "Suggere 3 de plus", disabled=visible_count >= max_display):
                st.session_state.suggestions_visible_count = min(visible_count + 3, max_display)
                st.rerun()
        
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
                    species    = data[key][0]["species"]
                    confidence = data[key][0]["confidence"]
                    color = confidence_color(confidence)

                    st.markdown(
                        f"<p style='font-size:1.4rem; font-weight:700; margin:0'>{_display_species_name(species)}</p>"
                        f"<p style='color:{color}; font-size:1.1rem; margin:4px 0 16px'>"
                        f"{confidence:.0%} confidence</p>",
                        unsafe_allow_html=True,
                    )

                    st.markdown("**Top 3**")
                    for rank, pred in enumerate(data[key], 1):
                        bar_pct = int(pred["confidence"] * 100)
                        st.markdown(f"**{rank}. {_display_species_name(pred['species'])}** — {pred['confidence']:.0%}")
                        st.progress(bar_pct)