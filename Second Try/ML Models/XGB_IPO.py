import pandas as pd
import numpy as np
import pickle


from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from imblearn.under_sampling import NearMiss

from sklearn.metrics import make_scorer
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report

df_mode=pd.read_pickle("/Users/klaus/Documents/Jarvis/Second Try/Data/df_mode_30.11")
df_fund=df_mode.loc[(df_mode["Y"]!=0) & (df_mode["Y"]!=4)].copy()
df_fund.loc[(df_mode["Y"]==1),"Y"]=0 #compare it to companies where nothing happened 
df_fund.loc[(df_mode["Y"]==2),"Y"]=1
df_fund.loc[(df_mode["Y"]==3),"Y"]=2
df_fund.reset_index(drop=True)
X=df_fund.drop(columns=["name","Y"])
y=df_fund["Y"]

sm = SMOTE(random_state = 2)
X,y = sm.fit_resample(X,y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

scorers = {
            'f1_score': make_scorer(f1_score, average='micro')
          }

params = { 'max_depth': [3, 5, 6, 10, 15, 20],
           'max_delta_step': [0,1,2,4],
           'learning_rate': [0.01, 0.1, 0.2, 0.3],
           'subsample': np.arange(0.5, 1.0, 0.1),
           'colsample_bytree': np.arange(0.4, 1.0, 0.1),
           'colsample_bylevel': np.arange(0.4, 1.0, 0.1),
           'n_estimators': [100, 500, 1000],
           'min_child_weight': [1,2,3,4]}
xgbr = xgb.XGBClassifier()
clf = RandomizedSearchCV(estimator=xgbr,
                         param_distributions=params,
                         scoring=scorers["f1_score"],
                         n_iter=20,
                         verbose=1,
                        )

clf.fit(X_train, y_train)
print("Best parameters:", clf.best_params_)
print("Lowest RMSE: ", (-clf.best_score_)**(1/2.0))

fix_params_ipo= clf.best_params_
fix_params_ipo["objective"] = 'multi:softproba'
fix_params_ipo["num_class"]=3

xgb_clf = xgb.XGBClassifier(**fix_params_ipo)
xgb_clf = xgb_clf.fit(X_train, y_train)
y_pred = xgb_clf.predict(X_test)
y_proba= xgb_clf.predict_proba(X_test)

output = open('IPO_xgb.pkl', 'wb')
pickle.dump(xgb_clf, output)
output.close()

print(classification_report(y_test, y_pred))
