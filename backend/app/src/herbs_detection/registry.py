from operator import le

from seaborn.objects import Path
from fastapi import Path
from sklearn.base import BaseEstimator
from herbs_detection.params import ALL_IMAGES_DIR, PROJECT_ROOT
#from herbs_detection import model
#from herbs_detection.params import MODEL_PATH, MODEL_NAME, ALIAS, MODEL_PATH, MODEL_MLFLOW_URI, PIPELINE_NAME, MODEL_NAME, MODEL_REGISTRY 
import pickle
import os
import mlflow

def saving_model(model: BaseEstimator, model_name: str) -> None:
    """Saves the model to disk"""
    # Save model weights (equivalent to model.save_weights() in Keras)
    save_dir = PROJECT_ROOT / "backend/app/models"
    save_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), save_dir / f"{model_name}.pt")

    # Save the LabelEncoder too — you need it to decode predictions back to species names
    with open(save_dir / f"{model_name}_label_encoder.pkl", "wb") as f:
        pickle.dump(le, f)

    print("Model saved.")

#    # Register in MLflow
#    mlflow.set_tracking_uri(mlflow_uri)
#    mlflow.set_experiment(pipeline_name)
#    with mlflow.start_run(run_name=model_name):
#        mlflow.log_artifact(str(save_dir / f"{model_name}.pt"), artifact_path="model")
#        mlflow.log_artifact(str(save_dir / f"{model_name}_label_encoder.pkl"), artifact_path="model")
#        mlflow.register_model(f"runs:/{mlflow.active_run().info.run_id}/model/{model_name}.pt", model_registry)

def load_model(model_name: str) -> BaseEstimator:
    """Loads the model from disk"""
    save_dir = PROJECT_ROOT / "backend/app/models"
    model_path = save_dir / f"{model_name}.pt"
    le_path = save_dir / f"{model_name}_label_encoder.pkl"

    if not model_path.exists() or not le_path.exists():
        raise FileNotFoundError("Model or label encoder not found. Train the model first.")

    # Recreate the architecture first (PyTorch requires this, unlike Keras)
    model = models.resnet18(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, NUM_CLASSES)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model = model.to(DEVICE)
    model.eval()

    with open(le_path, "rb") as f:
        le = pickle.load(f)

    print("Model loaded.")
    return model, le