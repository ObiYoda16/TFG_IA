import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RepeatedStratifiedKFold


import pandas as pd

import numpy as np
import os

import seaborn as sns
import matplotlib.pyplot as plt


class ClasificadorClasico:
    def __init__(self, classifier, name, x_data, y_data, columns):
        self.classifier = classifier
        self.name = name
        self.x_data = x_data
        self.y_data = y_data
        self.columns = columns
        self.params = {}
    
    #Busca los mejores valores para cada hiperpárametro de la array.
    def grid_fit(self, grid_params):
        cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)

        grid_search = GridSearchCV(estimator=self.classifier, 
                                param_grid=grid_params, 
                                cv=cv, n_jobs=-1, verbose=0, scoring = "f1_macro")

        results = grid_search.fit(self.x_data, self.y_data)        

        means = results.cv_results_['mean_test_score']
        params = results.cv_results_['params']
        self.classifier = grid_search.best_estimator_
        return zip(means, params)

    #Muestra los valores de un hiperpárametro
    def plot_result(self, values, y, title):
        fig, ax = plt.subplots(figsize=(15, 6))
        if not isinstance(values[0], str):
            ax.set_xticks(values)

        for i, element in enumerate(values):
            if element is None:
                values[i] = 'None'
                
        sns.lineplot(x = values, y = y, marker = 'o', ax=ax)
        plt.xlabel(title)
        plt.ylabel("Accuracy Score")
        plt.savefig(f"{self.name}/{title}{self.name}.png")
        plt

    #Ajusta el hiperparámetro y muestra sus mejores valores
    def fit_and_plot(self, new_param, values, plot):

        path = f"{self.name}"

        if not os.path.exists(path):
            os.mkdir(path) 

        self.params[new_param] = values
        zip_params = self.grid_fit(self.params)
        print()

        keys = {}
        keys['accuracy'] = []

        for mean, param in zip_params:
            for key in param:
                if key not in keys:
                    keys[key] = []
                keys[key].append(param[key])
            keys['accuracy'].append(mean)
        
        if plot:
            self.plot_result(values, keys['accuracy'], new_param)


        df = pd.DataFrame(keys)

        df = df.sort_values(by=['accuracy'], ascending=False)

        self.params[new_param] = [df[new_param].iloc[0]]
    
   
    def plot_feature_importance(self):
        cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)

        grid_search = GridSearchCV(estimator=self.classifier, 
                                param_grid=self.params, 
                                cv=cv, n_jobs=-1, verbose=1, scoring = "f1_macro")

        results = grid_search.fit(self.x_data, self.y_data)


        self.plot_feature_importance_aux(grid_search.best_estimator_.feature_importances_, self.columns)

    def plot_feature_importance_aux(self, importance, names):

        #Convertimos las listas a arrays de numpy
        feature_importance = np.array(importance)
        feature_names = np.array(names)

        #Creamos un df
        data={'feature_names': feature_names, 'feature_importance': feature_importance}
        fi_df = pd.DataFrame(data)

        #Ordenar segun la importancia
        fi_df.sort_values(by=['feature_importance'], ascending=False, inplace=True)

        #Tamaño
        plt.figure(figsize=(10,8))
        #Plot Searborn bar chart
        sns.barplot(x=fi_df['feature_importance'], y=fi_df['feature_names'])

        #Etiquetas
        plt.xlabel('FEATURE IMPORTANCE')
        plt.ylabel('FEATURE NAMES')

        plt.savefig(f"{self.name}/features_{self.name}.png")

    




