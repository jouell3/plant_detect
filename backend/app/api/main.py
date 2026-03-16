from fastapi import FastAPI


api = FastAPI()

@api.get("/")
def root():
    return {"message": "Hello World"}

api.post("/predict")
def predict():
    return {"message": "Prediction endpoint"}   

