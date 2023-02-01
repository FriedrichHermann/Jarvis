from fastapi import FastAPI
from pydantic import BaseModel
from app.model.model import predict_pipeline_funding
from app.model.model import predict_pipeline_IPO
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
    fundings_total:Optional[int] = 0
    country_name: Optional[str] = None # if both are none, the model will impute Germany 
    city_name: Optional[str] = None
    about: Optional[str] = ""
    patents_count:Optional[int]  = 0
        


@app.get("/")
def home():
    return{"health_check":"OK","model_version":model_version}

@app.post("/predict/funding")
def predict(Item: ListIn):
    items=Item.dict()
    a=predict_pipeline_funding(items)
    pred=np.around(a[0], 4).tolist()
    pred=list(pred)
    return {Item.name:pred}

@app.post("/predict/IPO")
def predict(Item: ListIn):
    items=Item.dict()
    a=predict_pipeline_IPO(items)
    pred=np.around(a[0], 4).tolist()

    return {Item.name:pred}