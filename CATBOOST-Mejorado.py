import catboost
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import make_scorer, f1_score
from sklearn.model_selection import RandomizedSearchCV
from metricas import metricas
from Clasificadores.procesamiento_audio import AudioProcessor
import numpy as np
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import make_scorer, f1_score
import catboost

processor = AudioProcessor(path_df="df_filter.csv", path_audio="audios/")
X_train = processor.X_train
y_train = processor.y_train
X_test = processor.X_test
y_test = processor.y_test



# Definir los parámetros de búsqueda
param_dist = {
    'iterations': [100, 200, 300, 400, 500],
    'depth': [4, 6, 8, 10],
    'learning_rate': [0.01, 0.05, 0.1, 0.15, 0.2],
    'l2_leaf_reg': [1, 3, 5, 7, 9],
    'border_count': [32, 64, 128]
}

# Crear el modelo CatBoost
model_catboost = catboost.CatBoostClassifier()

# Definir el scorer basado en f1-score
scorer = make_scorer(f1_score, average='weighted')

# Configurar RandomizedSearchCV
random_search = RandomizedSearchCV(estimator=model_catboost,
                                   param_distributions=param_dist,
                                   n_iter=50,  # Número de combinaciones a probar
                                   scoring=scorer,
                                   cv=5,  # Validación cruzada de 5 pliegues
                                   verbose=3,
                                   random_state=42,
                                   n_jobs=-1)

# Realizar la búsqueda de hiperparámetros
random_search.fit(X_train, y_train)

# Obtener los mejores hiperparámetros
best_params = random_search.best_params_
print(f"Best parameters found: {best_params}")

# Entrenar el modelo con los mejores hiperparámetros
best_model = random_search.best_estimator_
best_model.fit(X_train, y_train, eval_set=(X_test, y_test), verbose=True)

# Predecir y evaluar
y_pred = best_model.predict(X_test)
metricas = metricas(y_test, y_pred)
print(metricas)