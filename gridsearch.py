import torch
import pandas as pd
import numpy as np

import os
import glob

from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV

#####################--READ_DATA--##################################################
path_df = "./Clasificadores/df_filter.csv"
path_audio = "./audios/"

df = pd.read_csv(path_df)

df["file_cut_path"] = path_audio + df["file_name"]
df = df.reset_index(drop=True)
dataset = df["file_cut_path"]

#####################--READ_INFERENCE_TENSORS--########################################

log = []
for i in range (len(df)):
    log.append(torch.load("./torch_files14/torch_files14" + f"file{i}.pt"))

######################--SPLIT-DATASETS--##############################################
y = df["label"]
X_train, X_test, y_train, y_test = train_test_split(log, y, random_state = 42, stratify = y)

for i in range(len(X_train)):
    X_train[i] = X_train[i][0].detach().numpy()
    X_train[i] = np.concatenate(X_train[i])

for i in range(len(X_test)):
    X_test[i] = X_test[i][0].detach().numpy()
    X_test[i] = np.concatenate(X_test[i])

###################--GRIDSEARCHCV-#################################################

enc = LabelEncoder()
y_train_trans = enc.fit_transform(y_train)
y_test_trans = enc.fit_transform(y_test)

param_grid={
    'max_depth':[3,4,5],
    'learning_rate': [0.1, 0.01, 0.05],
    'gamma': [0, 0.25, 1.0],
    'reg_lamda': [0, 1.0, 10.0],
    'scale_pos_weight':[1, 3, 5] #XGBoost recommends sum(negative instances)/sum(positive instances)
}

optimal_params = GridSearchCV(
    estimator=XGBClassifier(objective='binary:logistic',
                                seed=42,
                                subsample=0.9,
                                colsample_bytree=0.5),
    param_grid=param_grid,
    scoring='roc_auc',
    verbose=0, #si quiero ver que hace el gridsearch verbose=2
    n_jobs= 10,
    cv = 3
)

optimal_params.fit(X_train, y_train_trans,
                   early_stopping_rounds=20, 
                   eval_metric="aucpr", 
                   eval_set=[(X_test,y_test_trans)],
                   verbose=False
                  )

print(optimal_params.best_params_)
