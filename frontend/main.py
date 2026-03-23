import streamlit as st

st.set_page_config(page_title="Plant Detect", layout="centered")

st.title("🌿 Plant Detect")
st.markdown("### Identification et gestion d'herbes aromatiques par IA")

st.divider()

st.markdown("""
Bienvenue sur **Plant Detect**, une application de reconnaissance d'herbes aromatiques
basée sur un modèle de deep learning (ResNet18) entraîné sur des images réelles.

---

### 📌 Que fait cette application ?

L'application permet d'identifier automatiquement une herbe aromatique à partir d'une photo,
et de gérer les données d'entraînement du modèle.

---

### 🔍 Prédiction

Accède à cet onglet pour :
- **Uploader une ou plusieurs images** de plantes
- **Prendre une photo** directement depuis ta caméra
- Obtenir une **identification automatique** de l'espèce avec un score de confiance
- Voir le **top 3** des espèces les plus probables

Le modèle est hébergé sur **Google Cloud Run** et les prédictions sont renvoyées en temps réel.

---

### 🏷️ Entraînement

Accède à cet onglet pour :
- **Parcourir les images** de ton dataset par dossier
- **Labelliser** chaque image (bonne qualité ou non) avant l'entraînement
- **Naviguer** par pages dans ton jeu de données

Les labels sont sauvegardés dans un fichier CSV réutilisable pour entraîner ou ré-entraîner le modèle.
""")

st.divider()
st.caption("Modèle : ResNet18 • Déploiement : Google Cloud Run • Stockage : Google Cloud Storage")
