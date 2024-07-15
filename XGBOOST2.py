import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from metricas import metricas
from Clasificadores.procesamiento_audio import AudioProcessor

from sklearn.model_selection import GridSearchCV, train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

file_path = './audio_features.csv'  # Ruta del archivo CSV
data = pd.read_csv(file_path)

# Crear la variable X excluyendo 'filename' y 'label'
X = data.drop(columns=['filename', 'label', 'pitch'])
nan_values = X.isna().sum()

# Mostrar solo las columnas con NaN
nan_columns = nan_values[nan_values > 0]
print(nan_columns)

# Crear la variable y que solo contenga 'label'
y = data['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42, stratify=y)




# Definir los parámetros para Grid Search
param_grid = {
    'max_bin': [8, 16, 32],
    'eta': [0.01, 0.1, 0.3],
    'max_depth': [3, 6, 9],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0],
}

# Inicialización del clasificador XGBoost
classifier = XGBClassifier(objective="binary:logistic", seed=42)

# Configuración de Grid Search con validación cruzada
grid_search = GridSearchCV(estimator=classifier, param_grid=param_grid, scoring='accuracy', cv=3, verbose=3)

# Entrenamiento del modelo utilizando Grid Search
grid_search.fit(X_train, y_train)

# Obtener el mejor clasificador después de Grid Search
best_classifier = grid_search.best_estimator_

# Predecir y evaluar el modelo
y_pred = best_classifier.predict(X_test)
score = accuracy_score(y_test, y_pred)
print(f"Accuracy: {score}")

# Mostrar otras métricas si están definidas en 'metricas'
metricas = metricas(y_test, y_pred)
print(metricas)
for i in range(len(metricas)):
    print(metricas[i])
