import streamlit as st

st.set_page_config(page_title="Plant Detect", layout="centered")

st.title("🌿 Plant Detect")
st.markdown("### Identification d'herbes aromatiques par IA")

st.divider()

st.markdown("""
Bienvenue sur **Plant Detect**, une application de reconnaissance d'herbes aromatiques
basée sur des modèles de deep learning (ResNet18 et EfficientNet) entraîné sur des images réelles.

---

### 📌 Que fait cette application ?

Cette application te permettra d'identifier une herbe aromatique à partir d'une simple photo prise avec ta caméra 
ou uploadée depuis tes dossiers. 
\n Elle te donne une prédiction en temps réel avec un score de confiance, et affiche le top 3 des espèces les plus probables.


---

Voici une petite description des différentes pages de l'application :

### 🔍 Prédiction

Accède à cet onglet pour :
- **Uploader une images** de plantes
- **Prendre une photo** directement depuis ta caméra
Ceci te permettra d'obtenir une **identification automatique** de l'espèce avec un score de confiance
Pour plus d'informations, le **top 3** des espèces les plus probables prédit par le modèle est également affiché.


\n Les différents modèles de reconnaissance d'herbes aromatiques ont été entraînés sur un dataset de 24000+ images réelles d'herbes aromatiques courantes prises dans des conditions variées (lumière, angles, arrière-plans).

##### Voici la liste complète des espèces reconnues par le modèle :

Angélique, Basilic, Bourrache, Camomille, Ciboulette, Coriandre, Aneth, Fenouil, Hysope, Lavande, Citronnelle, Verveine citronnée, Livèche, Menthe, Armoise, Origan, Persil, Romarin, Sauge, Sarriette, Estragon, Thym, Gaulthérie

\n Les images ont été obtenues via une API du site iNaturalist, qui regroupe des photos de plantes du monde entier avec des métadonnées de localisation et d'espèce.    
L'API est hébergée sur **Google Cloud Run** et les prédictions sont renvoyées en temps réel.

---

### 🏷️ Sélection d'Images

Accède à cet onglet pour :
- **Parcourir toutes les images** d'un dossier contenant des photos d'herbes aromatiques
- **Sélectionner** les différentes images (bonne qualité ou non) qui serviront à l'entraînement
- **Naviguer** par pages dans ton jeu de données

Le nom des images sélectionnées seront sauvegardés dans un fichier CSV réutilisable pour entraîner ou ré-entraîner le modèle.

### 📊 Prédiction par lot

Accède à cet onglet pour :
- **Uploader plusieurs images** à la fois
- **Visualiser les prédictions** obtenues à partir de deux modèles différents : 
    - un modèle PyTorch (ResNet18) 
    - un modèle Sklearn utilisant des features extraites d'un backbone EfficientNet B3 de 1536 dimensions.
""")

st.divider()
st.caption("""• Modèle : ResNet18 et EfficientNet B3  
           • Déploiement : Google Cloud Run  
           • Stockage : Google Cloud Storage    
           • Dataset : 24000+ images d'herbes aromatiques réelles (iNaturalist)     
           • Auteur : Jimmy OUELLET, Jaimes DE SOUSA GOMES, Thoams HEBERT, Edouard     
           • Code source : [GitHub](https://
           """)
