from fastapi import FastAPI, File, UploadFile
#from herbs_detection.load_data import load_model

api = FastAPI()

#model = load_model()

@api.get("/")
def root():
    return {"message": "Hello World"}

@api.post("/predict")
def predict(file: UploadFile = File(...)):
    return {"message": "Prediction endpoint"}   

