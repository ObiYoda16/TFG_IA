from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from metricas import metricas
from Clasificadores.procesamiento_audio import AudioProcessor
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
def getMejoresHiperParametros(X_train, y_train):
    param_grid = {
        'penalty': ['l2'],
        'C': [0.001],
        'solver': ['liblinear', 'newton-cg', 'lbfgs', 'sag', 'saga'],
        'max_iter': [100, 200, 500]
    }

    # Create a new logistic regression model
    model = LogisticRegression(max_iter=1000, verbose=1)

    # Use GridSearchCV to find the best hyperparameters
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='f1')
    grid_search.fit(X_train, y_train)

    # Print the best hyperparameters found
    print("Best Hyperparameters:", grid_search.best_params_)

processor = AudioProcessor(path_df="df_filter.csv", path_audio="audios/")
X_train = processor.X_train
y_train = processor.y_train
X_test = processor.X_test
y_test = processor.y_test

model= LogisticRegression()
#model= LogisticRegression(max_iter=1000,verbose=1, C=0.001, penalty="l2")
model.fit(X_train,y_train)

y_pred = model.predict(X_test)
score = accuracy_score(y_test, y_pred)
print(score)

metricas = metricas(y_test, y_pred)
print(metricas)
for i in range(len(metricas)):
    print(metricas[i])
"""
0.6454849498327759
              precision    recall  f1-score   support

         0.0       0.58      0.53      0.55       248
         1.0       0.69      0.73      0.71       350

    accuracy                           0.65       598
   macro avg       0.63      0.63      0.63       598
weighted avg       0.64      0.65      0.64       598
"""


