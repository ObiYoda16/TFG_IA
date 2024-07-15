from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from metricas import metricas
from Clasificadores.procesamiento_audio import AudioProcessor

processor = AudioProcessor(path_df="df_filter.csv", path_audio="audios/")
X_train   = processor.X_train
y_train  = processor.y_train
X_test = processor.X_test
y_test = processor.y_test


enc = LabelEncoder()
y_train_trans = enc.fit_transform(y_train)
y_test_trans = enc.fit_transform(y_test)


classifier = XGBClassifier(objective="binary:logistic", missing=None, seed=42,tree_method='gpu_hist')

classi = classifier.fit(X_train,y_train_trans, verbose=True, eval_metric="aucpr", eval_set=[(X_test,y_test_trans)])

y_pred = classi.predict(X_test)

score = accuracy_score(y_test_trans, y_pred)

print(score)
for i in range(len(metricas)):
    print(metricas[i])
"""    
            precision    recall  f1-score   support

 0             0.61      0.41      0.49       248
 1             0.66      0.81      0.73       350
 
accuracy                           0.65       598
macro avg      0.63      0.61      0.61       598
weighted avg   0.64      0.65      0.63       598
"""