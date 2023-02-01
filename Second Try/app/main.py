from fastapi import FastAPI
from pydantic import BaseModel
from app.model.model import predict_pipeline
from app.model.model import __version__ as model_version
import numpy as np
from typing import Optional

app=FastAPI()

from fastapi.exceptions import RequestValidationError



class ListIn(BaseModel):
    name:Optional[str]=""
    hasValidDomain:Optional[int] = 0 
    linkedin_url:Optional[str] = ""
    industry_name:Optional[str] = "" 
    founders_university:Optional[list] = []
    founders_background:Optional[list] = []
    investors:Optional[list] = []
    fundings_year:Optional[list] =[]
    total_funding:Optional[int] = 0
    fundings_month:Optional[list] =[]
    launch_year:Optional[int] = 2022
    fundings_total:Optional[int] = 0
    country_name: Optional[str] = None # if both are none, the model will impute Germany 
    city_name: Optional[str] = None
    about: Optional[str] = ""
    patents_count:Optional[int]  = 0
        


@app.get("/")
def home():
    return{"health_check":"OK","model_version":model_version}

@app.post("/predict")
def predict(Item: ListIn):
    items=Item.dict()
    a=predict_pipeline(items)
    pred=a[0].tolist()
    return {Item.name:pred}

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc):
    return PlainTextResponse(str(exc), status_code=400)