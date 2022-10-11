import json
import pickle
import re
from pathlib import Path

import pandas as pd
import requests
from fastai.data import *
from fastai.tabular import *
from fastai.tabular.all import *
from fastai.vision.all import *
from pandas.io.json import json_normalize
from requests.auth import HTTPBasicAuth
from sklearn.preprocessing import MultiLabelBinarizer

__version__="0.1.0"

BASE_DIR = Path(__file__).resolve(strict=True).parent

with open(f"{BASE_DIR}/xgb_model-{__version__}.pkl","rb") as f:
    model = pickle.load(f)



#implementing a predict function

#Step1.: Adjust incoming Data (Version 1 of Model. Simply give the Model a URL and Return a prediction)

def list_finder(companylist:list, api, fields="id,name,type,deleted,path,tagline,about,url,website_url,employees,employees_latest,corporate_industries,service_industries,technologies,growth_stage,traffic_summary,delivery_method,launch_year,launch_month,has_promising_founder,has_strong_founder,has_super_founder,total_funding,total_funding_source,last_funding,last_funding_source,company_status,last_updated,last_updated_utc,created_utc,similarweb_3_months_growth_unique,similarweb_6_months_growth_unique,similarweb_12_months_growth_unique,app_3_months_growth_unique,app_6_months_growth_unique,app_12_months_growth_unique,employee_3_months_growth_unique,employee_6_months_growth_unique,employee_6_months_growth_unique,employee_12_months_growth_unique,kpi_summary,team,investors,fundings,traffic,job_openings"):
    """
    will import the company information according to name, or website in the provided list. 
    Fields are set to "id,name,type,deleted,path,tagline,about,url,website_url,employees,employees_latest,corporate_industries,service_industries,technologies,growth_stage,traffic_summary,delivery_method,launch_year,launch_month,has_promising_founder,has_strong_founder,has_super_founder,total_funding,total_funding_source,last_funding,last_funding_source,company_status,last_updated,last_updated_utc,created_utc,similarweb_3_months_growth_unique,similarweb_6_months_growth_unique,similarweb_12_months_growth_unique,app_3_months_growth_unique,app_6_months_growth_unique,app_12_months_growth_unique,employee_3_months_growth_unique,employee_6_months_growth_unique,employee_6_months_growth_unique,employee_12_months_growth_unique,kpi_summary,team,investors,fundings,traffic,job_openings,exits,trading_multiple"
    by default. api key has to be provided in the api field. 
    
    Function will return the data in json as well as a list of names no information has been found for.
    """
    api_url="https://api.dealroom.co/api/v1/companies"
    API_KEY=api
    auth=HTTPBasicAuth(API_KEY, '')
    
    headers = {'Content-Type': 'application/json'}
    data_list=[]
    failed_names=[]
    
    for i in companylist:
        data={"keyword": "{}".format(i),
              "keyword_type": "website_domain",
              "keyword_match_type": "exact",
              "fields":fields,
              "limit": 1,
              "offset": 0
             }
        response = requests.post(api_url,data=json.dumps(data),auth=auth,headers=headers)
        data=response.json()
        if len(data["items"])==0:
            failed_names.append("{}".format(i))
        else:
            data_list.append(data["items"][0])
        
    return data_list


def predict_pipeline(searchlist:list,api):
    
    #Select only the fields the model uses:
    print("before")
    data=list_finder(searchlist,api) #has to be website URL
    print("after")
    df_json=json_normalize(data,sep="->")

    #1. One hot encode Technologies
    mlb = MultiLabelBinarizer()
    df = df_json.join(pd.DataFrame(mlb.fit_transform(df_json.pop('technologies')),
                               columns=mlb.classes_,
                               index=df_json.index))
    #2.Sin cos transform of Date

    #sin,cos transformation of date
    #transformation into datetime type
    dateadj_data=df.copy(deep=True)
    dateadj_data["last_updated_utc"] =  pd.to_datetime(df["last_updated_utc"])
    dateadj_data["kpi_summary->last_update_date_utc"] =  pd.to_datetime(df["kpi_summary->last_update_date_utc"])
    dateadj_data["created_utc"] =  pd.to_datetime(df["created_utc"])
    dateadj_data["launch_year"] =  pd.to_datetime(df["launch_year"],format="%Y").dt.year

    #new columns with cos and sin data for month. year will be same. dropping old ones
    dateadj_data["last_updated_month_cos"]=np.cos(2 * np.pi * dateadj_data["last_updated_utc"].dt.month/12.0)
    dateadj_data["last_updated_month_sin"]=np.sin(2 * np.pi * dateadj_data["last_updated_utc"].dt.month/12.0)
    dateadj_data["last_updated_year"]=dateadj_data["last_updated_utc"].dt.year

    dateadj_data["kpi_summary->last_update_date_utc_month_cos"]=np.cos(2 * np.pi * dateadj_data["kpi_summary->last_update_date_utc"].dt.month/12.0)
    dateadj_data["kpi_summary->last_update_date_utc_month_sin"]=np.sin(2 * np.pi * dateadj_data["kpi_summary->last_update_date_utc"].dt.month/12.0)
    dateadj_data["kpi_summary->last_update_date_utc_year"]=dateadj_data["kpi_summary->last_update_date_utc"].dt.year

    dateadj_data["created_utc_month_cos"]=np.cos(2 * np.pi * dateadj_data["created_utc"].dt.month/12.0)
    dateadj_data["created_utc_month_sin"]=np.sin(2 * np.pi * dateadj_data["created_utc"].dt.month/12.0)
    dateadj_data["created_utc_year"]=dateadj_data["created_utc"].dt.year

    dateadj_data.drop(columns=["last_updated_utc","kpi_summary->last_update_date_utc","created_utc"],inplace=True)


    #3. chosing the columns need for the model and applying the model

    cols=['type', 'about', 'employees', 'employees_latest', 'growth_stage', 'traffic_summary', 'launch_year', 'has_promising_founder', 'has_strong_founder', 'has_super_founder', 'total_funding', 'last_funding', 'company_status', 'last_updated_utc', 'created_utc', 'employee_3_months_growth_unique', 'job_openings', 'kpi_summary->last_update_date_utc', 'team->total', 'traffic->visitors', '3d technology', 'artificial intelligence', 'augmented reality', 'autonomous & sensor tech', 'big data', 'blockchain', 'computer vision', 'connected device', 'deep learning', 'deep tech', 'hardware', 'iot internetofthings', 'machine learning', 'mobile app', 'nanotech', 'natural language processing', 'quantum technologies', 'recognition technology', 'virtual reality']
    df_fin=dateadj_data[cols]

    cont_names, cat_names = cont_cat_split(y)


    y_to = TabularPandas(y, procs=[Categorify, FillMissing, Normalize], cat_names=cat_names, cont_names=cont_names, 
                                    )

    return model.predict(y_to.xs)


print(predict_pipeline(["apple.com"],api="66b3d061c986162ed7cbcb50a3f8e9b07d6a3aed"))

