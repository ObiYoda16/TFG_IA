
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import make_scorer, f1_score
from sklearn.model_selection import RandomizedSearchCV
from metricas import metricas
from Clasificadores.procesamiento_audio import AudioProcessor

from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
import numpy as np

# Asegúrate de que metricas y AudioProcessor están correctamente importados
from metricas import metricas  # Debes tener una implementación de metricas
from Clasificadores.procesamiento_audio import AudioProcessor  # Asegúrate que este módulo está correctamente implementado

# Load your audio features and labels
X = np.load("audio_features.npy")
y = np.load("audio_labels.npy")

# Assuming AudioProcessor processes and splits your data into train and test sets
processor = AudioProcessor(path_df="df_filter.csv", path_audio="audios/")
X_train = processor.X_train
y_train = processor.y_train
X_test = processor.X_test
y_test = processor.y_test

rf = RandomForestClassifier()
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)

metricas = metricas(y_test, y_pred)
for i in range(len(metricas)):
    print(metricas[i])
"""
0.6254180602006689
              precision    recall  f1-score   support

         0.0       0.57      0.39      0.46       248
         1.0       0.65      0.79      0.71       350

    accuracy                           0.63       598
   macro avg       0.61      0.59      0.59       598
weighted avg       0.62      0.63      0.61       598
"""
param_dist = {
    'n_estimators': [10, 50, 100, 200],
    'max_features': ['auto', 'sqrt', 'log2', None],
    'max_depth': [1, 5, 10, 20],
    'min_samples_split': [2, 5, 10, 20],
    'min_samples_leaf': [1, 5, 10, 20],
    'bootstrap': [True, False]
}

scoring = {
    'f1': make_scorer(f1_score)
}
rf_search = RandomizedSearchCV(
    rf,
    param_distributions=param_dist,
    scoring=scoring,
    refit='f1',
    n_iter=10,
    cv=5,
    verbose=2,
    n_jobs=-1
)

rf_search.fit(X_train, y_train)

best_rf = rf_search.best_estimator_
y_pred = best_rf.predict(X_test)
print("Mejores hiperparámetros:", rf_search.best_params_)

metricas = metricas(y_test, y_pred)
for i in range(len(metricas)):
    print(metricas[i])
