import streamlit as st

from i18n import get_language, render_language_selector

st.set_page_config(page_title="Plant Detect", layout="centered")

with st.sidebar:
    render_language_selector()

lang = get_language()

if lang == "fr":
    st.title("🌿 Plant Detect")
    st.markdown("### Identification d'herbes aromatiques par IA")

    st.divider()

    st.markdown("""
Bienvenue sur **Plant Detect**, une application de reconnaissance d'herbes aromatiques
basee sur des modeles de deep learning (ResNet18, EfficientNet B3 et TensorFlow) entraines sur des images reelles.

---

### 📌 Que fait cette application ?

Cette application permet d'identifier une herbe aromatique a partir d'une simple photo prise avec la camera du telephone
ou telechargee depuis un dossier.
\n Elle donne une prediction en temps reel avec un score de confiance, et affiche le top 3 des especes les plus probables pour differents modeles.


---

Voici une petite description des differentes pages de l'application :

### 🔍 Prediction d'herbes aromatiques

Accede a cet onglet pour :
- **Telecharger une images** de plantes aromatiques depuis un dossier present sur votre ordinateur
- Alternativement, il est possible de **Prendre une photo** directement depuis la camera du telephone
\n Ceci permettra d'obtenir une **identification automatique** de l'espece avec un score de confiance.
\n Il est egalement possible de visualiser le **top 3** des especes les plus probables predit par le modele.
\n Vous aurez aussi une suggestion de plats ou boissons possible a faire avec l'herbe aromatique identifiee, ainsi que des conseils de culture et d'entretien.
\n Un prompt de generation de recette a partir de l'herbe aromatique identifiee est egalement disponible.

""")
    st.caption("""Les differents modeles de reconnaissance d'herbes aromatiques ont ete entraines sur un dataset de 24000+ images reelles d'herbes aromatiques courantes prises dans des conditions variees (lumiere, angles, arriere-plans.
               Les images ont ete obtenues via une API du site iNaturalist, qui regroupe des photos de plantes du monde entier avec des metadonnees de localisation et d'espece.  """)

    st.markdown("""
##### Voici la liste complete des especes reconnues par le modele :

**Angelique, Basilic, Bourrache, Camomille, Ciboulette, Coriandre, Aneth, Fenouil, Hysope, Lavande, Citronnelle, Verveine citronnee, Liveche, Menthe, Armoise, Origan, Persil, Romarin, Sauge, Sarriette, Estragon, Thym, Gaultherie**

---

### 🍃 Detection de maladies sur les feuilles de tomates ou pommiers

Accede a cet onglet pour :

- **Telecharger une image** de feuille de tomate ou de pommier presentant des symptomes de maladies.
- **Obtenir une prediction** de la maladie presente sur la feuille, avec un score de confiance.
- **Visualiser les symptomes** associes a la maladie predite, ainsi que des conseils de traitement.
\n Les maladies reconnues sont : **Oidium du pommier, Pourriture noire du pommier, Rouille du pommier, Tache bacterienne de la tomate, Brulure precoce de la tomate, Tomate saine, Mildiou de la tomate, Mildiou foliaire de la tomate, Septoriose de la tomate, Tetranyque de la tomate, Tache cible de la tomate, Virus de la mosaique de la tomate, Virus du jaunissement en feuille de la tomate**.

### 📊 Prediction par lot (Multiples predictions d'aromates)

Accede a cet onglet pour :
- **Telecharger plusieurs images** a la fois
- Obtenir le top 3 des predictions pour chaque image ou faire un demande en bloc pour obtenir la premiere prediction de chaque image pour chaque modele.
- **Visualiser les predictions** obtenues a partir de differents modeles :
    - un modele PyTorch (ResNet18)
    - un modele Sklearn utilisant des features extraites d'un backbone EfficientNet B3 suivi d'une regression lineaire
    - un second modele PyTorch avec une architecture plus lourde
    - un modele TensorFlow

### 📊 Prediction par lot (Multiples predictions de maladies)
Accede a cet onglet pour :
- **Telecharger plusieurs images** de feuilles de tomates ou pommiers presentant des symptomes de maladies a la fois
- Obtenir le top 3 des predictions pour chaque image ou faire un demande en bloc pour obtenir la premiere prediction de chaque image pour chaque modele.
- **Visualiser les predictions** obtenues a partir de differents modeles :
    - un modele PyTorch (ResNet18)


### 🏷️ Selection d'Images (Image labelling)

Accede a cet onglet pour :
- **Parcourir toutes les images** d'un dossier contenant des photos.
- **Selectionner** les differentes images (bonne qualite) qui serviront a l'entrainement des differents modeles de reconnaissance d'herbes aromatiques.


Le nom des images selectionnees seront sauvegardes dans un fichier CSV reutilisable pour entrainer ou re-entrainer les modeles.
""")
    st.divider()
    st.caption("Les API qui permettent d'obtenir les predictions sont hebergees sur **Google Cloud Run** et les predictions sont renvoyees en temps reel.")

    st.divider()
    st.caption("""• Modele : ResNet18 et EfficientNet B3
               • Deploiement : Google Cloud Run
               • Stockage : Google Cloud Storage
               • Dataset : 24000+ images d'herbes aromatiques reelles (iNaturalist)
               • Auteurs : Jimmy OUELLET, Jaimes DE SOUSA GOMES, Thomas HEBERT, Edouard STEINER
               • Code source : [GitHub](https://github.com/jimmyouellet/plant-detect)
               """)
else:
    st.title("🌿 Plant Detect")
    st.markdown("### AI-Powered Aromatic Herb Identification")

    st.divider()

    st.markdown("""
Welcome to **Plant Detect**, an aromatic herb recognition application
powered by deep learning models (ResNet18, EfficientNet B3, and TensorFlow) trained on real-world images.

---

### 📌 What does this app do?

This app identifies an aromatic herb from a simple photo captured with a phone camera
or uploaded from a local folder.
\n It returns a real-time prediction with a confidence score, and displays the top 3 most likely species across multiple models.

---

Here is a quick overview of the app pages:

### 🔍 Aromatic Herb Prediction

Use this page to:
- **Upload a plant image** from a folder on your computer
- Or **take a photo** directly with your phone camera
\n This gives you an **automatic species identification** with a confidence score.
\n You can also view the **top 3** most likely species predicted by the model.
\n You also get recipe and drink suggestions for the detected herb, plus cultivation and care guidance.
\n A recipe prompt based on the detected herb is available as well.

""")
    st.caption("""The herb recognition models were trained on a dataset of 24,000+ real images of common aromatic herbs captured in varied conditions (lighting, angles, backgrounds).
               Images were collected through the iNaturalist API, which aggregates plant photos from around the world with species and location metadata.""")

    st.markdown("""
##### Full list of species recognized by the model:

**Angelica, Basil, Borage, Chamomile, Chives, Coriander, Dill, Fennel, Hyssop, Lavender, Lemongrass, Lemon Verbena, Lovage, Mint, Mugwort, Oregano, Parsley, Rosemary, Sage, Savory, Tarragon, Thyme, Wintergreen**

---

### 🍃 Disease Detection on Tomato or Apple Leaves

Use this page to:

- **Upload a tomato or apple leaf image** with potential disease symptoms.
- **Get a disease prediction** with confidence.
- **See symptoms and treatment guidance** related to the predicted disease.
\n Recognized diseases include: **Apple scab, Apple black rot, Cedar apple rust, Tomato bacterial spot, Tomato early blight, Tomato healthy, Tomato late blight, Tomato leaf mold, Tomato septoria leaf spot, Tomato spider mites, Tomato target spot, Tomato mosaic virus, Tomato yellow leaf curl virus**.

### 📊 Batch Prediction (Multiple Herb Predictions)

Use this page to:
- **Upload multiple images** at once
- Get top-3 predictions for each image, or submit a bulk request to get top-1 predictions per model.
- **Compare predictions** from multiple models:
    - a PyTorch model (ResNet18)
    - an sklearn model using EfficientNet B3 features followed by linear regression
    - a larger PyTorch model
    - a TensorFlow model

### 📊 Batch Prediction (Multiple Disease Predictions)
Use this page to:
- **Upload multiple tomato or apple leaf images** with disease symptoms at once
- Get top-3 predictions per image, or submit a bulk request for top-1 predictions.
- **Review predictions** generated by different models.

### 🏷️ Image Selection (Image Labeling)

Use this page to:
- **Browse all images** from a folder.
- **Select** high-quality images that will be used for model training.

Selected image names can be exported to a reusable CSV file for future training or retraining.
""")
    st.divider()
    st.caption("Prediction APIs are hosted on **Google Cloud Run** and return results in real time.")

    st.divider()
    st.caption("""• Model: ResNet18 and EfficientNet B3  
               • Deployment: Google Cloud Run  
               • Storage: Google Cloud Storage  
               • Dataset: 24,000+ real aromatic herb images (iNaturalist)  
               • Authors: Jimmy OUELLET, Jaimes DE SOUSA GOMES, Thomas HEBERT, Edouard STEINER  
               • Source code: [GitHub](https://github.com/jimmyouellet/plant-detect)  
               """)
