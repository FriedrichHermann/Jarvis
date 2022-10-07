import pandas as pd
import numpy as np
import random
from sklearn.metrics import auc, accuracy_score, confusion_matrix, mean_squared_error
from sklearn.model_selection import cross_val_score, GridSearchCV, KFold, RandomizedSearchCV, train_test_split
from fastai.tabular import *
from fastai.tabular.all import *
from fastai.vision.all import *
from fastai.data import *
import xgboost as xgb
import pickle

#Import the relevant data
df=pd.read_csv("/Users/klaus/Documents/Jarvis Data/Data/Clean_Data.csv")
df.drop(columns=["about"],inplace=True)

#Turn dependent variable to 0 and 1
df["Status"]=np.where((df["Status"]=="Rejected") | (df["Status"]=="To be rejected"), 0,1)

#Create Datablock from which training and test data can be extracted
cont_names, cat_names = cont_cat_split(df.drop(columns=["Status","name"]))
splits = RandomSplitter(valid_pct=0.2)(range_of(df))

to = TabularPandas(df, procs=[Categorify, FillMissing, Normalize], cat_names=cat_names, cont_names=cont_names, 
                                y_names="Status",y_block=CategoryBlock() ,splits=splits)

X_train, y_train = to.train.xs, to.train.ys.values.ravel()
X_test, y_test = to.valid.xs, to.valid.ys.values.ravel()

xgb_model = xgb.XGBClassifier(objective="binary:logistic", scale_pos_weight=8.399,random_state=42, colsample_bytree=0.9398611392758547, gamma= 0.4125663383463986, learning_rate= 0.08592149603644322, max_depth= 4, n_estimators= 113, subsample= 0.8092261699076477)
xgb_model.fit(X_train, y_train)


