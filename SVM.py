from sklearn import svm
from sklearn.metrics import accuracy_score
from metricas import metricas
from Clasificadores.procesamiento_audio import AudioProcessor

processor = AudioProcessor(path_df="df_filter.csv", path_audio="../audios/")
X_train = processor.X_train
y_train = processor.y_train
X_test = processor.X_test
y_test = processor.y_test

model = svm.SVC(verbose=True,C= 1, gamma= 'scale', kernel= 'rbf',decision_function_shape='ovr')
model.fit(X_train,y_train)
y_pred = model.predict(X_test)
score = accuracy_score(y_test, y_pred)
print(score)

metricas = metricas(y_test, y_pred)
print(metricas)
for i in range(len(metricas)):
    print(metricas[i])


"""
0.677257525083612
              precision    recall  f1-score   support

         0.0       0.68      0.42      0.52       248
         1.0       0.68      0.86      0.76       350

    accuracy                           0.68       598
   macro avg       0.68      0.64      0.64       598
weighted avg       0.68      0.68      0.66       598
"""
svc = svm.SVC()

# Definir la cuadrícula de hiperparámetros
param_grid = {
    'C': [0.1, 1, 10, 100],          # Parámetro de regularización
    'kernel': ['linear', 'rbf'],    # Tipo de kernel a probar
    'gamma': ['scale', 'auto']      # Coeficiente gamma para kernels rbf
}

# Definir el scorer (en este caso, exactitud para maximizar)
scorer = make_scorer(accuracy_score)

# Configurar la búsqueda grid con validación cruzada (5 folds)
grid_search = GridSearchCV(estimator=svc, param_grid=param_grid, scoring=scorer, cv=5, verbose=1)

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