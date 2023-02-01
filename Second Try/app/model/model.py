
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

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


from fuzzywuzzy import process
from fuzzywuzzy import fuzz

__version__="0.1.0"

BASE_DIR = Path(__file__).resolve(strict=True).parent

# Global Variables
currentMonth = datetime.now().month
currentYear = datetime.now().year

#Imputers and Encoders

with open(f"{BASE_DIR}/Encoders/Imputers/Industry_encoder.pkl","rb") as f:
    industry_enc = pd.read_pickle(f)
    
with open(f"{BASE_DIR}/Encoders/Imputers/Country_encoder.pkl","rb") as f:
    country_enc = pd.read_pickle(f)
    
#Transformation
    
with open(f"{BASE_DIR}/Encoders/Imputers/industries_dict.json","rb") as f:
    industries_dict = json.load(f)
    
#Scorer

with open(f"{BASE_DIR}/Encoders/univ_scorer.json", 'r') as fp:
    univ_scorer = json.load(fp)

with open(f"{BASE_DIR}/Encoders/inv_scorer.json", 'r') as fp:
    inv_scorer = json.load(fp)
    
#Datasets
    
with open(f"{BASE_DIR}/Data/df_mode.pkl","rb") as f:
    df_mode = pd.read_pickle(f)
    
with open(f"{BASE_DIR}/ML Models/funding_xgb_2.pkl","rb") as f:
    funding_xgb = pd.read_pickle(f)



# Functions that are used by following routines:
def translator(p):
    if isinstance(p,str):
        a=GoogleTranslator(source='de', target='en').translate(p)
        return a.lower().split(",")[0]
    else:
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
    if (isinstance(x["fundings_month"].iloc[0],np.ndarray) and len(x["fundings_month"].iloc[0])==0):
        if (isinstance(x["fundings_year"].iloc[0],np.ndarray) and len(x["fundings_year"].iloc[0])==0):
            a=currentYear*12-int(x["launch_year"].iloc[0])*12+currentMonth
            if ~x["total_funding"].isnull().iloc[0]:
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

#The Following function will implement the entire datatransformation routine
def data_cleaning(df):
    # 1. Industry Fields (str)
    df["industry_name"]=df["industry_name"].map(translator)
    df["industry_name"]=df["industry_name"].map(lambda p: industries_dict[p] if p in list(industries_dict.keys()) else None)
    
    # 2. Founders University Score (list)
    df["founders_university"]=list(map(lambda p: [similar(x,univ_scorer) for x in p] if isinstance(p,list) else [], df.founders_university))
    df["top_schools_score"]=df.founders_university.map(lambda p: sum([univ_scorer[x] for x in p if x in univ_scorer.keys()]) if isinstance(p,list) else 0)
    df["number_schools"]=df["founders_university"].map(lambda p: len(p))
    
    # 3 Investor Score (list)
    df["investors"]=list(map(lambda p: [similar(x,inv_scorer) for x in p] if isinstance(p,list) else 0, df.investors))
    a=map(lambda p: sum([inv_scorer[x] for x in p if x in inv_scorer.keys()]) if isinstance(p,list) else 0, df.investors)
    b=list(a)
    df["top_inv_score"]=b
    a=map(lambda p: len(p) if isinstance(p,list) else p, df.investors)
    b=list(a)
    df["investors_total"]=b
    
    # 4 Avg time funding (fundings_month: np.ndarray, fundings_year: np.ndarray, launch_year: int)
    df["fundings_year"]=df["fundings_year"].map(lambda p: np.array(p) if isinstance(p,list)  else p)
    df["fundings_month"]=df["fundings_month"].map(lambda p: np.array(p) if isinstance(p,list)  else p)
    df["avg_time_funding"]=0
    indexers=df[df["launch_year"].apply(lambda p: False if (isinstance(p,list)) else True)].index
    for i in indexers:
        df.loc[i,"avg_time_funding"]=avg_fund_func(df.iloc[i:i+1])
        
    # 5 Is Bootstrapped
    for i in df.columns:
        df[i]=df[i].map(lambda p: None if ((isinstance(p,list) and len(p)==0) or (isinstance(p,np.ndarray) and len(p)==0)) else p)
    df["is_bootstrapped"]=0
    total_funding_bol=(df["total_funding"].apply(lambda p: True if (str(p)=="nan" or str(p)=="None") else False ))
    investors_total_bol=(df["investors_total"].apply(lambda p: True if (str(p)=="nan" or str(p)=="None") else False))
    fundings_year_bol=(df["fundings_year"].apply(lambda p: True if (str(p)=="nan" or str(p)=="None") else False))

    df.loc[total_funding_bol&investors_total_bol&fundings_year_bol,"is_bootstrapped"]=1
    
    
    for i in df.columns:
        if isinstance(df[i][0],np.ndarray):
            if df[i][0].size == 0:
               df[i][0]==None 
        elif (df[i][0]=="" or df[i][0]==0 or df[i][0]==[]):
            df[i][0]==None 
    df.loc[df["is_bootstrapped"]==1,"total_funding"]=0
    df.loc[df["is_bootstrapped"]==1,"avg_time_funding"]=0
    df.loc[df["is_bootstrapped"]==1,"fundings_total"]=0
    df.loc[df["is_bootstrapped"]==1,"investors"]=0
    df["missing_values"]=df.isnull().sum(axis=1)
    
    return df
    
def imputer(df):
    # Impute total_funding
    df_mice = df_mode.filter(['total_funding', 'fundings_total', 'launch_year',"avg_time_funding","number_top_schools","is_bootstrapped"], axis=1)
    to_be_imp_ind=df.loc[df["is_bootstrapped"]!=1].index
    to_be_imp=df.loc[df["is_bootstrapped"]!=1]
    to_be_imp=to_be_imp.filter(['total_funding', 'fundings_total', 'launch_year',"avg_time_funding","number_top_schools","is_bootstrapped"], axis=1)
    df_mice=pd.concat([df_mice,to_be_imp],ignore_index=True)
    needed_rows=to_be_imp.shape[0]
    mice_imputer = IterativeImputer(estimator=KNeighborsRegressor(n_neighbors=5), n_nearest_features=None, imputation_order='ascending', missing_values=np.nan, min_value = 0.0)
    df_mice_imputed = pd.DataFrame(mice_imputer.fit_transform(df_mice), columns=df_mice.columns)
    df.loc[to_be_imp_ind,"total_funding"]=df_mice_imputed.tail(n=needed_rows)["total_funding"].values
    
    #Impute total_fundings
    if (df["fundings_total"][0]==0) & (df["total_funding"][0]!=0):
        median_funding=np.median(df_mode.loc[df_mode["is_bootstrapped"]!=1,"total_funding"])
        def upper_bound(p):
            if p<=5:
                return p
            else:
                return 5

        df.loc[df["fundings_total"]==0,"fundings_total"]=upper_bound(np.round(df.loc[df["fundings_total"]==0,"total_funding"].values/median_funding))
    
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
    
    
    
    # Transformer of Country
    
    
    # About
    stop_words=set(stopwords.words('english'))
    a=map(lambda p: tokenizer(p) if isinstance(p,str) else 0, df["about"]) # remove stopwords and seperate str into list of words
    df["about"]=list(a)

    a=map(lambda p: len([w.lower() for w in p if w.isalpha()]) if isinstance(p,list) else p, df["about"]) # remove all special signs and non alphabetic characters
    df["about"]=list(a)
    
    # Domain
    df["website_url"]=df["hasValidDomain"].map(lambda p: p[0] if isinstance(p,list) else p)
    df["linkedin_url"]=df["linkedin_url"].map(lambda p: 1 if isinstance(p,str) else 0)
    
    #Launch Year
    mean_year=np.mean(df.loc[~(df["launch_year"]<=0),"launch_year"])
    df.loc[(df["launch_year"]<=0),"launch_year"]=mean_year
    df["launch_year"]=currentYear-df["launch_year"]
    return df[['name','about','website_url','linkedin_url','total_funding','patents_count','launch_year',
               'investors_total','fundings_total','country_name','industry_name',
               'top_inv_score','top_schools_score','number_schools','avg_time_funding','is_bootstrapped','missing_values']]
    
    
    
def predict_pipeline(input_dict:dict):
    df=pd.json_normalize(input_dict)
    df=data_cleaning(df)
    df=imputer(df)
    a=funding_xgb.predict_proba(df.drop(columns=["name"]))
    return a