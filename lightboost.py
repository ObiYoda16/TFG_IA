import catboost
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import make_scorer, f1_score, accuracy_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn import svm
from sklearn.metrics import accuracy_score
from metricas import metricas
from Clasificadores.procesamiento_audio import AudioProcessor
from metricas import metricas
from Clasificadores.procesamiento_audio import AudioProcessor

###################--CLASIFICADOR LIGHTBOOST- #################################################
import lightgbm as lgb
processor = AudioProcessor(path_df="df_filter.csv", path_audio="audios/")
X_train = processor.X_train
y_train = processor.y_train
X_test = processor.X_test
y_test = processor.y_test

X_train_lgb = np.array(X_train)
y_train_lgb = np.array(y_train)
X_test_lgb  = np.array(X_test)
y_test_lgb  = np.array(y_test)

enc = LabelEncoder()
y_train_trans = enc.fit_transform(y_train_lgb )
y_test_trans  =  enc.fit_transform(y_test_lgb)

train_data_lgb = lgb.Dataset(X_train_lgb, label=y_train_trans)
validation_data_lgb = lgb.Dataset(X_test_lgb, label=y_test_trans)

#model = lgb.LGBMClassifier(num_leaves=7,min_data_in_leaf=100,max_depth=10,min_gain_to_split=0.5,num_iterations=100)
model = lgb.LGBMClassifier()
model.fit(X_train_lgb, y_train_lgb)
y_pred = model.predict(X_test)
acuraccy = accuracy_score(y_pred, y_test)
print(acuraccy)

metricas = metricas(y_test, y_pred)
print(metricas)
for i in range(len(metricas)):
    print(metricas[i])
"""
0.6471571906354515
              precision    recall  f1-score   support

         0.0       0.61      0.42      0.49       248
         1.0       0.66      0.81      0.73       350

    accuracy                           0.65       598
   macro avg       0.64      0.61      0.61       598
weighted avg       0.64      0.65      0.63       598
"""