import io
import json
import os
from pathlib import Path

import requests
import streamlit as st
from loguru import logger
from PIL import Image

API_URL = os.environ.get("API_URL", "https://plant-detect-backend-649164185154.europe-west1.run.app")


_FICHES_PATH = Path(__file__).parent.parent / "fiches.json"
FICHES: dict = json.loads(_FICHES_PATH.read_text(encoding="utf-8")) if _FICHES_PATH.exists() else {}

st.set_page_config(page_title="Plant Predictor", layout="wide")

st.title("🌿 Plant Predictor")
st.markdown("Vous pouvez soit choisir une image dans vos dossiers, soit prendre une photo directement avec votre caméra. Le modèle de reconnaissance d'herbes aromatiques vous donnera une prédiction en temps réel avec un score de confiance. N'hésitez pas à tester plusieurs images pour voir les résultats !")
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

st.divider()

if st.button("🔍 Identify", type="primary", use_container_width=False):
    with st.spinner("Analysing…"):
        try:
            logger.info("predict_herb | file={}", uploaded_file.name)
            response = requests.post(
                f"{API_URL}/predict_herb",
                files={"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type or "image/jpeg")},
                timeout=60,
            )
            response.raise_for_status()

        except requests.exceptions.ConnectionError:
            st.error("Cannot reach the API. Check that the service is running.")
            logger.error("API connection error | url={}", API_URL)
            st.stop()
        except requests.exceptions.HTTPError as e:
            st.error(f"API error: {e}")
            logger.error("API HTTP error | {}", e)
            st.stop()

    # ── Parse results ─────────────────────────────────────────────────────
    data = response.json()  # {predictions: [{species, confidence}, ...]}
    st.text(data)
    
    top3 = data["predictions"]

    # ── Display ───────────────────────────────────────────────────────────
    st.subheader("Résultats")
    col_img, col_res = st.columns([1, 2])

    with col_img:
        img = Image.open(io.BytesIO(uploaded_file.getvalue()))
        st.image(img, use_column_width=True, caption=uploaded_file.name)

    with col_res:
        species    = top3[0]["species"]
        confidence = top3[0]["confidence"]
        color = "#2e7d32" if confidence >= 0.75 else "#f57c00" if confidence >= 0.5 else "#c62828"

        st.markdown(
            f"<p style='font-size:1.4rem; font-weight:700; margin:0'>{species}</p>"
            f"<p style='color:{color}; font-size:1.1rem; margin:4px 0 16px'>"
            f"{confidence:.0%} confidence</p>",
            unsafe_allow_html=True,
        )

        st.markdown("**Top 3**")
        for rank, pred in enumerate(top3, 1):
            bar_pct = int(pred["confidence"] * 100)
            st.markdown(f"**{rank}. {pred['species']}** — {pred['confidence']:.0%}")
            st.progress(bar_pct)

    # ── Herb info card ─────────────────────────────────────────────────────
    fiche = FICHES.get(top3[0]["species"].lower())
    if fiche:
        st.divider()
        nom_fr_md = f"[{fiche['nom_fr']}]({fiche['wikipedia_fr']})" if fiche.get("wikipedia_fr") else fiche['nom_fr']
        nom_en_md = f"[{fiche['nom_en']}]({fiche['wikipedia_en']})" if fiche.get("wikipedia_en") else fiche['nom_en']
        st.markdown(f"### À propos — {nom_fr_md} (*{nom_en_md}*)")
        st.markdown(fiche["description"])

        info_col1, info_col2 = st.columns(2)
        with info_col1:
            st.markdown(f"🌸 **Arôme** : {fiche['arome']}")
            st.markdown(f"🌱 **Culture** : {fiche['culture']}")
            st.markdown(f"⚠️ **Toxicité** : {fiche['toxicite']}")
        with info_col2:
            st.markdown(f"🍽️ **Usages** : {', '.join(fiche['usages'])}")
            st.markdown(f"🤝 **Compatible avec** : {', '.join(fiche['compatibilites'])}")


