from sklearn.base import BaseEstimator
#from herbs_detection import model
#from herbs_detection.params import MODEL_PATH, MODEL_NAME, ALIAS, MODEL_PATH, MODEL_MLFLOW_URI, PIPELINE_NAME, MODEL_NAME, MODEL_REGISTRY 
import pickle
import os
import mlflow

mlflow.set_tracking_uri(MODEL_MLFLOW_URI)

def save_model(model: BaseEstimator, 
               name: str = PIPELINE_NAME, 
               MODEL_REGISTRY: str = MODEL_REGISTRY) -> None:
    
    """Save the model to the specified path."""
        
        # Implement the logic to save the model (e.g., using pickle, joblib, etc.)

    if MODEL_REGISTRY == "mlflow": 
        mlflow.sklearn.log_model(f"models:/{PIPELINE_NAME}@{ALIAS}")
    elif MODEL_REGISTRY == "local": 
        if not os.path.exists(MODEL_PATH) : 
            os.mkdir(MODEL_PATH)
        with open(os.path.join(MODEL_PATH, f'{name}.pkl'),"wb") as f:
            pickle.dump(model, f)
    
        

def load_model(name: str = PIPELINE_NAME, 
               MODEL_REGISTRY: str = MODEL_REGISTRY) -> BaseEstimator:
    """Load the model from the specified path."""
    # Implement the logic to load the model (e.g.os.nameing pickle, joblib, etc.)
    if MODEL_REGISTRY == "mlflow": 
        model = mlflow.sklearn.load_model(f"models:/{PIPELINE_NAME}@{ALIAS}")
    elif MODEL_REGISTRY == "local": 
        with open(os.path.join(MODEL_PATH, f"{name}.pkl"), "rb") as f:
            model = pickle.load(f)
    return model



def load_model_mkflow() -> BaseEstimator:
    """Load the model from the specified path using mkflow."""
    model = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}@{ALIAS}")
    return model