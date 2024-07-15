import keras
import numpy as np
import tensorflow as tf
from keras import layers
from sklearn.metrics import classification_report

from Clasificadores.procesamiento_audio import AudioProcessor
print(tf.config.list_physical_devices('GPU'))
processor = AudioProcessor(path_df="./df_filter.csv", path_audio="../audios/")
X_train = processor.X_train
y_train = processor.y_train
X_test = processor.X_test
y_test = processor.y_test

X_train_formated = np.array(X_train).reshape(-1, 152576, 1)
y_train_formated = np.array(y_train).reshape(-1, 1)
X_test_formated = np.array(X_test).reshape(-1, 152576, 1)
y_test_formated = np.array(y_test).reshape(-1, 1)

y_train_formated = keras.utils.to_categorical(y_train_formated)
y_test_formated = keras.utils.to_categorical(y_test_formated)

num_classes = 2
input_shape = (152576, 1)

model = keras.Sequential([
    layers.Flatten(input_shape=input_shape),  # Aplanar la entrada para usar capas densas
    layers.Dense(64, activation="relu",),  # Primera capa densa con 128 neuronas
    layers.Dropout(0.25),
    layers.Dense(32, activation="relu"),  # Segunda capa densa con 64 neuronas
    layers.Dropout(0.5),
    layers.Dense(16, activation="relu"),  # Segunda capa densa con 64 neuronasºexit

    layers.Dense(num_classes, activation="softmax"),  # Capa de salida con activación softmax
])

loss = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
optimizer = keras.optimizers.Adam(learning_rate=1e-5)

model.compile(loss=loss, optimizer=optimizer, metrics=["accuracy"])
model.summary()
# Train the model
model.fit(X_train_formated, y_train_formated, epochs=10, batch_size=2, validation_split=0.2, verbose=1)

# Predict and evaluate
y_pred = model.predict(X_test_formated, verbose=1)
y_pred_bool = np.argmax(y_pred, axis=1)
loss, accuracy = model.evaluate(X_test_formated, y_test_formated)
print(f'Loss: {loss}, Accuracy: {accuracy}')
print(classification_report(np.argmax(y_test_formated, axis=1), y_pred_bool))
"""
              precision    recall  f1-score   support

           0       0.00      0.00      0.00       248
           1       0.59      1.00      0.74       350

    accuracy                           0.59       598
   macro avg       0.29      0.50      0.37       598
weighted avg       0.34      0.59      0.43       598
"""
"""
    layers.Flatten(input_shape=input_shape),  # Aplanar la entrada para usar capas densas
    layers.Dense(64, activation="relu"),  # Primera capa densa con 128 neuronas
    layers.Dense(32, activation="relu"),  # Segunda capa densa con 64 neuronas
    layers.Dropout(0.5),
    layers.Dense(num_classes, activation="softmax"),  # Capa de salida con activación softmax
Loss: 0.8041754364967346, Accuracy: 0.6521739363670349
              precision    recall  f1-score   support

           0       0.61      0.46      0.52       248
           1       0.67      0.79      0.73       350

    accuracy                           0.65       598
   macro avg       0.64      0.62      0.62       598
weighted avg       0.65      0.65      0.64       598
"""