
from deep_translator import GoogleTranslator
from datetime import datetime
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.neighbors import KNeighborsRegressor
from geopy.geocoders import Nominatim
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from pathlib import Path
import numpy as np
import pandas as pd
import json 
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


from fuzzywuzzy import process
from fuzzywuzzy import fuzz

__version__="0.1.4"

BASE_DIR = Path(__file__).resolve(strict=True).parent

# Global Variables
currentMonth = datetime.now().month
currentYear = datetime.now().year

#Imputers and Encoders

with open(f"{BASE_DIR}/Encoders/Imputers/Industry_encoder.pkl","rb") as f:
    industry_enc = pd.read_pickle(f)
    
with open(f"{BASE_DIR}/Encoders/Imputers/Country_encoder.pkl","rb") as f:
    country_enc = pd.read_pickle(f)

with open(f"{BASE_DIR}/Encoders/Imputers/lbin_ind.pkl","rb") as f:
    industry_onehot = pd.read_pickle(f)
    
with open(f"{BASE_DIR}/Encoders/Imputers/lbin_count.pkl","rb") as f:
    country_onehot = pd.read_pickle(f)
#Transformation
    
with open(f"{BASE_DIR}/Encoders/Imputers/industries_dict.json","rb") as f:
    industries_dict = json.load(f)
    
with open(f"{BASE_DIR}/Encoders/Imputers/scaler.pkl","rb") as f:
    scaler = pd.read_pickle(f)
    
#Scorer

with open(f"{BASE_DIR}/Encoders/univ_scorer.json", 'r') as fp:
    univ_scorer = json.load(fp)

with open(f"{BASE_DIR}/Encoders/inv_scorer.json", 'r') as fp:
    inv_scorer = json.load(fp)
    
with open(f"{BASE_DIR}/Encoders/backgr_scorer.json", 'r') as fp:
    backgr_scorer = json.load(fp)
    
#Datasets
    
with open(f"{BASE_DIR}/Data/df_mode.pkl","rb") as f:
    df_mode = pd.read_pickle(f)
    
with open(f"{BASE_DIR}/Data/needed_rows.pkl","rb") as f:
    needed_rows = pd.read_pickle(f)
    
with open(f"{BASE_DIR}/Data/testy.pkl","rb") as f:
    testy = pd.read_pickle(f)
    
with open(f"{BASE_DIR}/Data/testX.pkl","rb") as f:
    testX = pd.read_pickle(f)
    
    
with open(f"{BASE_DIR}/ML Models/cat_modl.pkl","rb") as f:
    funding_model = pd.read_pickle(f)
    
with open(f"{BASE_DIR}/ML Models/IPO_xgb.pkl","rb") as f:
    IPO_xgb = pd.read_pickle(f)

with open(f"{BASE_DIR}/ML Models/0_1_xgb.pkl","rb") as f:
    nulleins_xgb = pd.read_pickle(f)


# Functions that are used by following routines:
def translator(p):
    if isinstance(p,str):
        a=GoogleTranslator(source='de', target='en').translate(p)
        return a.lower().split(",")[0]
    else:
        None

def lower_split(p):
    try:
        words=[]
        for i in range(len(p)):
            words.extend(p[i].lower().split(","))
        
        return words
    except:
        None

        
def industry_conv(p):
    try:
        entry=[]
        for i in range(len(p)):
            for j in range(industries_df.shape[0]):
                if len(p)==1 and p[i]==industries_df["Name"].iloc[j]:
                    entry.append(industries_df["Category"].iloc[j])
                else:
                    None
        if len(entry)==0:
            for j in range(industries_df.shape[0]):
                if p[0]==industries_df["Name"].iloc[j]:
                    entry.append(industries_df["Category"].iloc[j])                       
        return pd.unique(entry)[0]
    except:
        None

def similar(x,dicti) : 
    if process.extractOne(x.lower(), dicti.keys(), scorer=fuzz.token_sort_ratio)[1]>=90: 
        return process.extractOne(x.lower(), dicti.keys(), scorer=fuzz.token_sort_ratio)[0]

def avg_time(x,y):
    x=x.iloc[0]
    if isinstance(x,int):
        avg_fund=x-y
    elif isinstance(x,np.ndarray):
        dif=np.array(x[0:len(x)-1])-np.array(x[1:len(x)])
        avg_fund=(np.mean(np.append(dif,x[len(x)-1]-y))) # will add time to first funding to even it out with companies that had one funding
    else:
        avg_fund=0

    return avg_fund

def avg_fund_func(x):
    if x["fundings_month"].isnull().iloc[0]:
        if x["fundings_year"].isnull().iloc[0]:
            a=currentYear*12-int(x["launch_year"].iloc[0])*12+currentMonth
            if ~x["number_fundings"].isnull().iloc[0]:
                a=a/2
            else:
                None
        else:
            a=avg_time(x["fundings_year"]*12,int(x["launch_year"])*12)
    else:
        try:
            z=x["fundings_year"]*12+x["fundings_month"]
        except:
            z=x["fundings_year"]*12
        a=avg_time(z,int(x["launch_year"])*12)
    return a/12

def country_getter(p):
    geolocator = Nominatim(user_agent = "geoapiExercises")
    location = geolocator.geocode(p["city_name"])
    return location.address.split(",")[-1].replace(" ","")

def tokenizer(x:str):
    stop_words = set(stopwords.words('english'))
    tokens=word_tokenize(x)
    tokens = [w for w in tokens if w not in stop_words]
    return tokens
# Variables

cont_cols=['about',
 'total_funding',
 'patents_count',
 'launch_year',
 'investors_total',
 'number_fundings',
 'top_inv_score',
 'top_schools_score',
 'number_schools',
 'avg_time_funding',
 'backgr_score',
 'missing_values'] # gives all the continous fields in the dataset
#The Following function will implement the entire datatransformation routine
def data_cleaning(df):
    # 1. Industry Fields (str)
    df["industry_name"]=df["industry_name"].map(lambda p: p.lower().split(",")[0])
    df["industry_name"]=df["industry_name"].map(translator)
    
    
    # 2. Founders University Score (list)
    df["founders_university"]=list(map(lambda p: [similar(x,univ_scorer) for x in p] if isinstance(p,list) else [], df.founders_university))
    df["top_schools_score"]=df.founders_university.map(lambda p: sum([univ_scorer[x] for x in p if x in univ_scorer.keys()]) if isinstance(p,list) else 0)
    df["number_schools"]=df["founders_university"].map(lambda p: len(p) if (len(p)>0 and str(p[0])!="None") else 0)
    
    # 2. Founders Background Score (list)
    df["founders_background"]=list(map(lambda p: [similar(x,backgr_scorer) for x in p] if isinstance(p,list) else [], df.founders_background))
    df["backgr_score"]=df.founders_background.map(lambda p: sum([backgr_scorer[x] for x in p if x in backgr_scorer.keys()]) if isinstance(p,list) else 0)

    
    # 3 Investor Score (list)
    df["investors"]=list(map(lambda p: [similar(x,inv_scorer) for x in p] if isinstance(p,list) else 0, df.investors))
    a=map(lambda p: sum([inv_scorer[x] for x in p if x in inv_scorer.keys()]) if isinstance(p,list) else 0, df.investors)
    b=list(a)
    df["top_inv_score"]=b
    a=map(lambda p: len(p) if (len(p)>0 and str(p[0])!="None") else 0, df.investors)
    b=list(a)
    df["investors_total"]=b
    
        
    for i in [a for a in list(df.columns) if a not in ["avg_time_funding","city_name","patents_count"]]:
        if isinstance(df[i][0],np.ndarray):
            if (df[i][0].size == 0):
               df[i][0]=None 
        elif (df[i][0]==""):
            df[i][0]=None 
        elif (df[i][0]==0):
            df[i][0]=None 
        elif isinstance(df[i][0],list):
            if (len(df[i][0]) == 0):
               df[i][0]=None
        
    # 5 Is Bootstrapped
    for i in df.columns:
        df[i]=df[i].map(lambda p: None if ((isinstance(p,list) and len(p)==0) or (isinstance(p,np.ndarray) and len(p)==0)) else p)
    df["is_bootstrapped"]=0
    total_funding_bol=(df["total_funding"].apply(lambda p: True if (str(p)=="nan" or str(p)=="None") else False ))
    number_fundings_bol=(df["number_fundings"].apply(lambda p: True if (str(p)=="nan" or str(p)=="None") else False ))
    investors_total_bol=(df["investors_total"].apply(lambda p: True if (str(p)=="nan" or str(p)=="None") else False))
    fundings_year_bol=(df["fundings_year"].apply(lambda p: True if (str(p)=="nan" or str(p)=="None") else False))

    df.loc[total_funding_bol&investors_total_bol&fundings_year_bol&number_fundings_bol,"is_bootstrapped"]=1
    
    df["fundings_year"]=df["fundings_year"].map(np.array)
    df["fundings_month"]=df["fundings_month"].map(np.array)
    
    # 7 Missing Values
    df.loc[df["is_bootstrapped"]==1,"total_funding"]=0
    df.loc[df["is_bootstrapped"]==1,"avg_time_funding"]=0
    df.loc[df["is_bootstrapped"]==1,"number_fundings"]=0
    df.loc[df["is_bootstrapped"]==1,"investors"]=0
    df["missing_values"]=df.isnull().sum(axis=1)
    
    return df
    
    
def imputer(df):
    #industries
    df["industry_name"]=df["industry_name"].map(lambda p: industries_dict[p] if p in list(industries_dict.keys()) else None)   
    
    # Impute patents count
    df.loc[df["patents_count"].apply(lambda p: True if (str(p) == "None" or str(p)=="nan") else False),"patents_count"]=0
        # Impute Country + city_name
    if (df["country_name"].isna()[0])&(~df["city_name"].isna()[0]):
        country_indexers=df.loc[(df["country_name"].isna())&(~df["city_name"].isna())].index
        for i in country_indexers:
            df.loc[i,"country_name"]=country_getter(df.iloc[i])
    else:
        df["country_name"]="Germany"

    if (~df["country_name"].isna()[0]):
        df["country_name"]=df["country_name"].map(lambda p: translator(p).title())
        label=country_enc.transform(df.country_name)
        df["country_name"]=label
        
    df=df.join(pd.DataFrame(country_onehot.transform(df["country_name"].values.reshape(-1,1)),columns=needed_rows[104-57:]+["nan_0"]))
    df=df.join(pd.DataFrame(industry_onehot.transform(df["industry_name"].values.reshape(-1,1)),columns=needed_rows[104-57-31:(104-57)]+["nan_1"]))
    
    # About
    stop_words=set(stopwords.words('english'))
    a=map(lambda p: tokenizer(p) if isinstance(p,str) else 0, df["about"]) # remove stopwords and seperate str into list of words
    df["about"]=list(a)

    a=map(lambda p: len([w.lower() for w in p if w.isalpha()]) if isinstance(p,list) else p, df["about"]) # remove all special signs and non alphabetic characters
    df["about"]=list(a)
    
    # Domain
    df["website_url"]=df["hasValidDomain"].map(lambda p: p[0] if isinstance(p,list) else p)
    df["linkedin_url"]=df["linkedin_url"].map(lambda p: 1 if (isinstance(p,str) and len(p)!=0) else 0)
 
    wanted_for_imp=['launch_year','about','patents_count','top_schools_score','number_schools','backgr_score','top_inv_score','investors_total','is_bootstrapped','missing_values',"total_funding"]
     # Impute total_funding
    if (df["total_funding"].isnull()[0] and df["is_bootstrapped"]!=1):
        t=df["investors_total"]
        df["total_funding"]=np.median(df_mode.loc[(df_mode["number_fundings"]!=0)&(df_mode["investors_total"]==t)&(df_mode["is_bootstrapped"]==0),"total_funding"])

     
    if (df["number_fundings"].isnull()[0] and df["is_bootstrapped"]!=1):
        t=df["investors_total"]
        df["number_fundings"]=np.median(df_mode.loc[(df_mode["number_fundings"]!=0)&(df_mode["investors_total"]==t)&(df_mode["is_bootstrapped"]==0),"number_fundings"])

    # 4 Avg time funding (fundings_month: np.ndarray, fundings_year: np.ndarray, launch_year: int)
    df["avg_time_funding"]=0
    df["avg_time_funding"]=avg_fund_func(df)
    
    #Launch Year
    mean_year=np.mean(df.loc[~(df["launch_year"]<=0),"launch_year"])
    df.loc[(df["launch_year"]<=0),"launch_year"]=mean_year
    df["launch_year"]=currentYear-df["launch_year"]
    if df["launch_year"][0]==0:
        df["launch_year"]+=0.5

    #needed columns for prediciton
    df=df[needed_rows]
    # scaling data (min/max)
    df=df.fillna(0)
    data=df[cont_cols]
    data_scaled=scaler.transform(data)
    df[cont_cols]=data_scaled
    return df


    
def predict_pipeline_funding(input_dict:dict):
    df=pd.json_normalize(input_dict)
    df=data_cleaning(df)
    df=imputer(df)
    a=(funding_model.predict_proba(df.drop(columns=["name"]))[:,1] >= input_dict["threshold"]).astype(int)
    return a

def recall_precision(threshold):
    y_pred=(funding_model.predict_proba(testX)[:,1] >= threshold).astype(int)
    recall=recall_score(testy,y_pred)
    precision=precision_score(testy,y_pred)
    return {"recall":recall,"precision":precision,"threshold":threshold}