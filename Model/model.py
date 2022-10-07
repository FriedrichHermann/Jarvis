import pickle
import re
from pathlib import Path

import pandas as pd
import requests
import json
from pandas.io.json import json_normalize
from requests.auth import HTTPBasicAuth

from sklearn.preprocessing import MultiLabelBinarizer

__version__="0.1.0"

BASE_DIR = Path(__file__).resolve(strict=True).parent

with open(f"{BASE_DIR}/xgb_model-{__version__}.pkl","rb") as f:
    model = pickle.load(f)



#implementing a predict function
def list_finder(companylist:list, api, fields="id,name,type,deleted,path,tagline,about,url,website_url,employees,employees_latest,corporate_industries,service_industries,technologies,growth_stage,traffic_summary,delivery_method,launch_year,launch_month,has_promising_founder,has_strong_founder,has_super_founder,total_funding,total_funding_source,last_funding,last_funding_source,company_status,last_updated,last_updated_utc,created_utc,similarweb_3_months_growth_unique,similarweb_6_months_growth_unique,similarweb_12_months_growth_unique,app_3_months_growth_unique,app_6_months_growth_unique,app_12_months_growth_unique,employee_3_months_growth_unique,employee_6_months_growth_unique,employee_6_months_growth_unique,employee_12_months_growth_unique,kpi_summary,team,investors,fundings,traffic,job_openings,exits,trading_multiple"):
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

a=open("/Users/klaus/Documents/Jarvis Data/Data/data (1).json")
data=list_finder(searchlist) #has to be website URL
df_json=json_normalize(data,sep="->")


mlb = MultiLabelBinarizer()
df = df_json.join(pd.DataFrame(mlb.fit_transform(df_json.pop('technologies')),
                               columns=mlb.classes_,
                               index=df_json.index))

df.drop(columns=["name","Status"])

def predict_pipeline(text)


cont_names, cat_names = cont_cat_split(y)


y_to = TabularPandas(y, procs=[Categorify, FillMissing, Normalize], cat_names=cat_names, cont_names=cont_names, 
                                 )

xgb_model.predict(y_to.xs)