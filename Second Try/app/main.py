from fastapi import FastAPI
from pydantic import BaseModel
from app.model.model import predict_pipeline
from app.model.model import __version__ as model_version
import numpy as np
from typing import Optional

app=FastAPI()

from fastapi.exceptions import RequestValidationError



class ListIn(BaseModel):
    name:str 
    hasValidDomain:int = 0 
    linkedin_url:str = ""
    industry_name:str = "" 
    founders_university:list = []
    investors:list = []
    fundings_year:list =[]
    total_funding:int = 0
    fundings_month:list =[]
    launch_year:int = 2022
    fundings_total:int = 0
    country_name: Optional[str] = None
    city_name: Optional[str] = None
    about: str = ""
    patents_count:int  = 0
        


@app.get("/")
def home():
    return{"health_check":"OK","model_version":model_version}

@app.post("/predict")
def predict(Item: ListIn):
    items=Item.dict()
    a=predict_pipeline(items)
    pred=a[0].tolist()
    return {"Prediction":pred}

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc):
    return PlainTextResponse(str(exc), status_code=400)