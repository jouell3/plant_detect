# Plant Detect

API de détection d'herbes aromatiques basée sur ResNet18 (PyTorch), déployée sur Google Cloud Run. Les fichiers modèles sont stockés sur Google Cloud Storage et téléchargés au démarrage du conteneur.

---

## Prérequis

- Python 3.11+
- Docker Desktop
- [Google Cloud SDK](https://cloud.google.com/sdk/docs/install) (`gcloud`)
- Un projet GCP : `bootcamparomatic`

### Authentification GCP

```bash
gcloud auth login
gcloud auth application-default login
gcloud config set project bootcamparomatic
```

---

## Structure

```
backend/
  app/
    api/         # FastAPI routes
    models/      # fichiers modèles locaux (gitignorés)
    src/
      herbs_detection/
        model.py   # chargement modèle + inférence
notebooks/
  training_pytorch.ipynb  # entraînement + upload vers GCS
docker/
  backend.Dockerfile
scripts/
  setup_gcs_bucket.sh
```

---

## 1. Bucket GCS — setup

Le bucket héberge les fichiers `resnet18_plants.pt` et `label_encoder.pkl`.

```bash
bash scripts/setup_gcs_bucket.sh
```

Ce script :
- Crée le service account `plant-detect-sa`
- Crée le bucket `plant-detect-models` en `europe-west1`
- Active le versioning
- Accorde les droits `objectAdmin` au service account
- Attache le service account au service Cloud Run (si déjà déployé)

> Variables configurables en tête du script : `GCP_PROJECT`, `GCS_BUCKET_NAME`, `GCP_REGION`, `CLOUD_RUN_SERVICE`, `GCP_SERVICE_ACCOUNT`

---

## 2. Entraînement et upload du modèle

Ouvrir `notebooks/training_pytorch.ipynb` et exécuter toutes les cellules dans l'ordre.

La dernière cellule upload automatiquement `resnet18_plants.pt` et `label_encoder.pkl` vers `gs://plant-detect-models/models/`.

---

## 3. Développement local

### Lancer l'API sans Docker

```bash
make serve
```

### Lancer avec Docker (authentification GCP via ADC)

```bash
make run
```

Cela monte `~/.config/gcloud/application_default_credentials.json` dans le conteneur pour l'auth GCS.

### Tester l'API

```bash
make test
# ou manuellement :
curl -X POST http://localhost:8080/predict_herb \
  -F "file=@data/raw/all_images/dill_0.jpg"
```

---

## 4. Tests

### Installation des dépendances de test

```bash
cd backend
pip install -r requirements-dev.txt
```

### Lancer tous les tests

```bash
cd backend
pytest -v
```

### Tests disponibles

| Fichier | Ce qui est testé |
|---|---|
| `tests/test_api.py` | Endpoints FastAPI (`/`, `/predict_herb`, `/predict_illness`, `/predict-set`, `/predict-set_illness`) |
| `tests/test_deploy_script.py` | Fonctions du script `scripts/deploy_models.py` (`_pick_latest`, `build_sklearn_files`, `upload`) |

> Les tests ne chargent aucun poids de modèle et ne font aucun appel à GCS — toutes les fonctions ML sont remplacées par des stubs légers.

---

## 5. Déploiement sur GCP

### 4.1 Créer le repository Artifact Registry (une seule fois)

```bash
gcloud artifacts repositories create plant-detect \
  --repository-format=docker \
  --location=europe-west1 \
  --project=bootcamparomatic
```

Authentifier Docker :

```bash
gcloud auth configure-docker europe-west1-docker.pkg.dev
```

### 4.2 Builder et pusher l'image

```bash
make build_gcp
```

Cela build l'image pour `linux/amd64` et la push vers Artifact Registry.

### 4.3 Premier déploiement Cloud Run

```bash
gcloud run deploy plant-detect-backend \
  --image=europe-west1-docker.pkg.dev/bootcamparomatic/plant-detect/plant-detect-backend \
  --platform=managed \
  --region=europe-west1 \
  --memory=2Gi \
  --service-account=plant-detect-sa@bootcamparomatic.iam.gserviceaccount.com \
  --set-env-vars GCS_BUCKET_NAME=plant-detect-models,GCS_MODELS_PREFIX=models,GCS_PROJECT=bootcamparomatic \
  --allow-unauthenticated
```

> `--memory=2Gi` est requis pour charger PyTorch + ResNet18.

### 4.4 Mises à jour suivantes

Après un nouvel entraînement, rebuilder et pusher suffit — Cloud Run déploie automatiquement la nouvelle image :

```bash
make build_gcp
```

---

## 6. Mettre à jour la configuration du service

### Changer les variables d'environnement

```bash
gcloud run services update plant-detect-backend \
  --set-env-vars GCS_BUCKET_NAME=plant-detect-models,GCS_MODELS_PREFIX=models \
  --region=europe-west1
```

### Changer le service account

```bash
gcloud run services update plant-detect-backend \
  --service-account plant-detect-sa@bootcamparomatic.iam.gserviceaccount.com \
  --region=europe-west1
```

### Augmenter la mémoire

```bash
gcloud run services update plant-detect-backend \
  --memory=2Gi \
  --region=europe-west1
```

---

## 7. Logs

```bash
make get_log_gcp
```

Ou en direct :

```bash
gcloud run services logs tail plant-detect-backend \
  --region=europe-west1 \
  --project=bootcamparomatic
```

---

## Variables d'environnement

| Variable | Défaut | Description |
|---|---|---|
| `GCS_BUCKET_NAME` | `plant-detect-models` | Bucket GCS contenant les modèles |
| `GCS_MODELS_PREFIX` | `models` | Dossier dans le bucket |
| `GCS_PROJECT` | `bootcamparomatic` | Projet GCP |
| `MODEL_PATH` | `/tmp/plant_models` | Cache local des fichiers modèles |
| `GOOGLE_APPLICATION_CREDENTIALS` | — | Chemin vers le fichier ADC (local uniquement) |
