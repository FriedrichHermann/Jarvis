from fastapi import FastAPI
from pydantic import BaseModel
from app.model.model import predict_pipeline
from app.model.model import __version__ as model_version

app=FastAPI()

class ListIn(BaseModel):
    list: list

class PredictionOut(BaseModel):
    prediction: list

@app.get("/")
def home():
    return{"health_check":"OK","model_version":model_version}

@app.post("/predict",response_model=PredictionOut)
def predict(payload: ListIn):
    score = predict_pipeline(payload.list,"66b3d061c986162ed7cbcb50a3f8e9b07d6a3aed")
    return {"score": score}