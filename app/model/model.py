import json
import pickle
import re
from pathlib import Path

import pandas as pd
import requests
import fastai
from fastai.data import *
from fastai.tabular import *
from fastai.tabular.all import *
from fastai.vision.all import *
from pandas import json_normalize
from requests.auth import HTTPBasicAuth
from sklearn.preprocessing import MultiLabelBinarizer

__version__="0.1.0"

BASE_DIR = Path(__file__).resolve(strict=True).parent

with open(f"{BASE_DIR}/xgb_model-{__version__}.pkl","rb") as f:
    model = load_learner(f)



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

def extractor(json_dic):
    """
    Function takes in json data and transforms it to dataframe. Values in dictionary form will not be transformed
    """
    df_1=json_normalize(json_dic,sep="->")
    for i in list(df_1.columns):
        try:
            if pd.json_normalize(json_dic,record_path="{}".format(i)).empty:
                df_1=df_1
            else:
                df_2=pd.json_normalize(json_dic[0],record_path="{}".format(i),meta=["id"],meta_prefix="{}->".format(i), sep=",") 
            
            #problem: Will also turn normal lists into DF with column name 0
            #new if clause will rename those columns to original name:
                if df_2.columns[0]==0:
                    df_2.rename(columns={df_2.columns[0]: ''.format(i)}, inplace=True)
                else:
                    None
                
            df_1=pd.merge(df_1,df_2, left_on="id",right_on="{}->id".format(i),how="left",suffixes=('', '->{}'.format(i)))
            df_1.drop("{}".format(i),axis=1,inplace=True)
        except:
            df_1=df_1
        else:
            df_1=json_normalize(json_dic[0],sep="->") 
            
    return df_1


def predict_pipeline(searchlist:list,api):
    try:
        #Select only the fields the model uses:
        data=list_finder(searchlist,api) #has to be website URL
        df=extractor(data)


        #1. chosing the columns need for the model and applying the model

        cols=['name', 'type', 'employees', 'employees_latest', 'growth_stage', 'traffic_summary', 'launch_year', 'has_promising_founder', 'has_strong_founder', 'has_super_founder', 'total_funding', 'last_funding', 'company_status', 'last_updated_utc', 'created_utc', 'employee_3_months_growth_unique', 'job_openings']
        df=df[cols]

        row, cls, probs  = model.predict(df.iloc[0]) 
        a=[df.iloc[0]["name"], cls, probs]
    except:
        a="your URL has resulted in no return"
    return a



print(predict_pipeline(["dog.com"],"66b3d061c986162ed7cbcb50a3f8e9b07d6a3aed"))