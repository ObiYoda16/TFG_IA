import pandas as pd
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.svm import SVC
from sklearn.metrics import make_scorer, accuracy_score, classification_report
from sklearn import svm
from sklearn.metrics import accuracy_score
from metricas import metricas
from Clasificadores.procesamiento_audio import AudioProcessor

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



# Definir el modelo SVC
svc = SVC()

# Definir la cuadrícula de hiperparámetros
param_grid = {
    'C': [0.1, 1],          # Parámetro de regularización
    'kernel': ['linear', 'rbf'],    # Tipo dK
    'gamma': ['scale', 'auto']      # Coeficiente gamma para kernels rbf
}

# Definir el scorer (en este caso, exactitud para maximizar)
scorer = make_scorer(accuracy_score)

# Configurar la búsqueda grid con validación cruzada (5 folds)
grid_search = GridSearchCV(estimator=svc, param_grid=param_grid, scoring=scorer, cv=5, verbose=2)

# Entrenar el modelo usando búsqueda grid
grid_search.fit(X_train, y_train)

# Mostrar los mejores hiperparámetros encontrados
print("Mejores hiperparámetros encontrados:")
print(grid_search.best_params_)

# Obtener el mejor modelo entrenado
best_model = grid_search.best_estimator_

# Predecir con el mejor modelo
y_pred = best_model.predict(X_test)

# Calcular la exactitud en el conjunto de prueba
accuracy = accuracy_score(y_test, y_pred)
print(f"Exactitud en conjunto de prueba: {accuracy}")

# Mostrar métricas de clasificación detalladas
print("Reporte de clasificación detallado:")
print(classification_report(y_test, y_pred))
