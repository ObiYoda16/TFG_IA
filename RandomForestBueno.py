import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import make_scorer, f1_score
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from metricas import metricas
from Clasificadores.procesamiento_audio import AudioProcessor
file_path = './audio_features.csv'  # Ruta del archivo CSV
data = pd.read_csv(file_path)

# Crear la variable X excluyendo 'filename' y 'label'
X = data.drop(columns=['filename', 'label'])
nan_values = X.isna().sum()

# Mostrar solo las columnas con NaN
nan_columns = nan_values[nan_values > 0]
print(nan_columns)

# Crear la variable y que solo contenga 'label'
y = data['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
rf = RandomForestClassifier()
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)

metricas1 = metricas(y_test, y_pred)
for i in range(len(metricas1)):
    print(metricas1[i])
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
    'max_features': [ 'sqrt', 'log2', None],
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

metricas1 = metricas(y_test, y_pred)
for i in range(len(metricas1)):
    print(metricas1[i])
