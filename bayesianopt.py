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

from bayes_opt import BayesianOptimization, UtilityFunction
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from xgboost import XGBClassifier


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

dtrain = xgb.DMatrix(X_train, label=y_train_trans)

def bo_tune_xgb(max_depth, gamma ,learning_rate):
    params = {'max_depth': int(max_depth),
              'gamma': gamma,
              'learning_rate':learning_rate,
              'subsample': 0.8,
              'eta': 0.1,
              'eval_metric': 'rmse'}
    #Cross validating with the specified parameters in 5 folds and 70 iterations
    cv_result = xgb.cv(params, dtrain, num_boost_round=20, nfold=5)
    #Return the negative RMSE
    return -1.0 * cv_result['test-rmse-mean'].iloc[-1]

xgb_bo = BayesianOptimization(bo_tune_xgb, {'max_depth': (3, 10),
                                             'gamma': (0, 1),
                                             'learning_rate':(0,1)
                                            })


utility = UtilityFunction(kind="ei")
xgb_bo.maximize(n_iter=1, init_points=8, acquisition_function=utility)


from sklearn.metrics import classification_report

params = xgb_bo.max['params']
print(params)

params['max_depth']= int(params['max_depth'])

classifier2 = XGBClassifier(**params).fit(X_train, y_train_trans)

y_pred = classifier2.predict(X_test)

def metricas(objetivo, prediccion):
    matriz_conf = confusion_matrix(objetivo, prediccion)
    score = accuracy_score(objetivo, prediccion)
    reporte = classification_report(objetivo, prediccion)
    metricas = [matriz_conf, score, reporte]
    return(metricas)

metricas = metricas(y_test_trans, y_pred)

for i in range(len(metricas)):
    print(metricas[i])
