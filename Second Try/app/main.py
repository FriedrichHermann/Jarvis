from fastapi import FastAPI
from pydantic import BaseModel
from app.model.model import predict_pipeline_funding
from app.model.model import recall_precision
from app.model.model import __version__ as model_version
import numpy as np
from typing import Optional

app=FastAPI()




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
    number_fundings:Optional[int] = 0
    country_name: Optional[str] = None # if both are none, the model will impute Germany 
    city_name: Optional[str] = None
    about: Optional[str] = ""
    patents_count:Optional[int]  = 0
    threshold:Optional[float] = 0.4
        


@app.get("/")
def home():
    return{"health_check":"OK","model_version":model_version}

@app.post("/predict/funding")
def predict(Item: ListIn):
    items=Item.dict()
    a=predict_pipeline_funding(items)
    ret=a[0].tolist()
    return {Item.name:ret}

@app.post("/precision_recall_score")
def score(Item: ListIn):
    a=recall_precision(Item.threshold)
    return a
 

#@app.post("/predict/0_1")
#def predict(Item: ListIn):
    items=Item.dict()
    a=predict_pipeline_nulleins(items)
    a=a[0].tolist()
    pred=[round(p,5) for p in a]

    return {Item.name:pred}

#@app.post("/recall_precision")
#def predict(Item: ListIn):
    items=Item.dict()
    nulleins=predict_pipeline_nulleins(items)
    nulleins=nulleins[0].tolist()
    if nulleins[0]>=0.5:
        a=predict_pipeline_funding(items)
        a=a[0].tolist()
        pred=[round(p,5) for p in a]
    else:
        a=predict_pipeline_IPO(items)
        a=a[0].tolist()
        pred=[round(p,5) for p in a]
    return {Item.name:pred}