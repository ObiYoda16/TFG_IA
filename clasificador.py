from Clasificadores.procesamiento_audio import AudioProcessor

##################################################################################
processor = AudioProcessor(path_df="Clasificadores/df_filter.csv", path_audio="./audios/")
X_train = processor.X_train
y_train = processor.y_train
X_test = processor.X_test
y_test = processor.y_test


"""





###################--CLASIFICADOR CATBOOST- #################################################
import catboost

train_data = catboost.Pool(data=X_train, label=y_train)
model_catboost = catboost.CatBoostClassifier(iterations=500, depth=6, learning_rate=0.1, loss_function='Logloss')
model_catboost.fit(train_data, eval_set=(X_test, y_test), verbose=True)
y_pred = model_catboost.predict(X_test)
metricas = metricas(y_test, y_pred)
print(metricas)


"""
