import streamlit as st

st.set_page_config(page_title="Plant Detect", layout="centered")

st.title("🌿 Plant Detect")
st.markdown("### Identification d'herbes aromatiques par IA")

st.divider()

st.markdown("""
Bienvenue sur **Plant Detect**, une application de reconnaissance d'herbes aromatiques
basée sur des modèles de deep learning (ResNet18, EfficientNet B3 et TensorFlow) entraîné sur des images réelles.

---

### 📌 Que fait cette application ?

Cette application permet d'identifier une herbe aromatique à partir d'une simple photo prise avec la caméra du téléphone 
ou téléchargée depuis un dossier. 
\n Elle donne une prédiction en temps réel avec un score de confiance, et affiche le top 3 des espèces les plus probables pour différents modèles.


---

Voici une petite description des différentes pages de l'application :

### 🔍 Prédiction d'herbes aromatiques

Accède à cet onglet pour :
- **Télécharger une images** de plantes aromatiques depuis un dossier present sur votre ordinateur
- Alternativement, il est possible de **Prendre une photo** directement depuis la caméra du téléphone  
\n Ceci permettra d'obtenir une **identification automatique** de l'espèce avec un score de confiance.  
\n Il est également possible de visualiser le **top 3** des espèces les plus probables prédit par le modèle.   
\n Vous aurez aussi une sugegstion de plats ou boissons possible à faire avec l'herbe aromatique identifiée, ainsi que des conseils de culture et d'entretien.  
\n Un prompt de génération de recette à partir de l'herbe aromatique identifiée est également disponible.

""")
st.caption("""Les différents modèles de reconnaissance d'herbes aromatiques ont été entraînés sur un dataset de 24000+ images réelles d'herbes aromatiques courantes prises dans des conditions variées (lumière, angles, arrière-plans.   
           Les images ont été obtenues via une API du site iNaturalist, qui regroupe des photos de plantes du monde entier avec des métadonnées de localisation et d'espèce.  """)

st.markdown("""
##### Voici la liste complète des espèces reconnues par le modèle :

**Angélique, Basilic, Bourrache, Camomille, Ciboulette, Coriandre, Aneth, Fenouil, Hysope, Lavande, Citronnelle, Verveine citronnée, Livèche, Menthe, Armoise, Origan, Persil, Romarin, Sauge, Sarriette, Estragon, Thym, Gaulthérie**

---

### 🍃 Détection de maladies sur les feuilles de tomates ou pommiers

Accède à cet onglet pour :

- **Télécharger une image** de feuille de tomate ou de pommier présentant des symptômes de maladies.
- **Obtenir une prédiction** de la maladie présente sur la feuille, avec un score de confiance.
- **Visualiser les symptômes** associés à la maladie prédite, ainsi que des conseils de traitement.
\n Les maladies reconnues sont : **Oïdium du pommier, Pourriture noire du pommier, Rouille du pommier, Tache bactérienne de la tomate, Brûlure précoce de la tomate, Tomate saine, Mildiou de la tomate, Mildiou foliaire de la tomate, Septoriose de la tomate, Tétranyque de la tomate, Tache cible de la tomate, Virus de la mosaïque de la tomate, Virus du jaunissement en feuille de la tomate**.

### 📊 Prédiction par lot (Multiples prédictions d'aromates)

Accède à cet onglet pour :
- **Télécharger plusieurs images** à la fois
- Obtenir le top 3 des prédictions pour chaque image ou faire un demande en bloc pour obtenir la première prédiction de chaque image pour chaque modèle.
- **Visualiser les prédictions** obtenues à partir de différents modèles : 
    - un modèle PyTorch (ResNet18) 
    - un modèle Sklearn utilisant des features extraites d'un backbone EfficientNet B3 suivi d'une regression linéaire
    - un second modèle PyTorch avec une architecture plus lourde
    - un modèle TensorFlow

### 📊 Prédiction par lot (Multiples prédictions de maladies)
Accède à cet onglet pour :
- **Télécharger plusieurs images** de feuilles de tomates ou pommiers présentant des symptômes de maladies à la fois
- Obtenir le top 3 des prédictions pour chaque image ou faire un demande en bloc pour obtenir la première prédiction de chaque image pour chaque modèle.
- **Visualiser les prédictions** obtenues à partir de différents modèles : 
    - un modèle PyTorch (ResNet18) 

 
### 🏷️ Sélection d'Images (Image labelling)

Accède à cet onglet pour :
- **Parcourir toutes les images** d'un dossier contenant des photos.
- **Sélectionner** les différentes images (bonne qualité) qui serviront à l'entraînement des différents modèles de reconnaissance d'herbes aromatiques.


Le nom des images sélectionnées seront sauvegardés dans un fichier CSV réutilisable pour entraîner ou ré-entraîner les modèles.
""")
st.divider()
st.caption("Les API qui permettent d'obtenir les prédictions sont hébergées sur **Google Cloud Run** et les prédictions sont renvoyées en temps réel.")

st.divider()
st.caption("""• Modèle : ResNet18 et EfficientNet B3  
           • Déploiement : Google Cloud Run  
           • Stockage : Google Cloud Storage    
           • Dataset : 24000+ images d'herbes aromatiques réelles (iNaturalist)     
           • Auteurs : Jimmy OUELLET, Jaimes DE SOUSA GOMES, Thomas HEBERT, Edouard STEINER    
           • Code source : [GitHub](https://github.com/jimmyouellet/plant-detect)
           """)
