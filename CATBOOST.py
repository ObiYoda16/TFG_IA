import catboost
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import make_scorer, f1_score
from sklearn.model_selection import RandomizedSearchCV
from metricas import metricas
from Clasificadores.procesamiento_audio import AudioProcessor

processor = AudioProcessor(path_df="df_filter.csv", path_audio="audios/")
X_train = processor.X_train
y_train = processor.y_train
X_test = processor.X_test
y_test = processor.y_test

train_data = catboost.Pool(data=X_train, label=y_train)
#model_catboost = catboost.CatBoostClassifier(iterations=100, depth=4, learning_rate=0.1, loss_function='Logloss')
model_catboost = catboost.CatBoostClassifier()
model_catboost.fit(train_data, eval_set=(catboost.Pool(data=X_test, label=y_test)), verbose=True)
y_pred = model_catboost.predict(X_test)
metricas = metricas(y_test, y_pred)
print(metricas)
"""
              precision    recall  f1-score   support

           0      0.64      0.44      0.52       248
           1      0.67      0.83      0.74       350

    accuracy                          0.66       598
   macro avg      0.66      0.63      0.63       598
weighted avg      0.66      0.66      0.65       598
"""