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

# Encode labels
enc = LabelEncoder()
y_train_trans = enc.fit_transform(y_train)
y_test_trans = enc.fit_transform(y_test)

# Initialize and train the classifier
classifier = XGBClassifier(objective="binary:logistic", missing=None, seed=42)
classi = classifier.fit(X_train, y_train_trans, verbose=True, eval_metric="aucpr", eval_set=[(X_test, y_test_trans)])

# Predict and evaluate
y_pred = classi.predict(X_test)
score = accuracy_score(y_test_trans, y_pred)
print(f"Accuracy: {score}")

# Print detailed classification report
print(classification_report(y_test_trans, y_pred, target_names=['Class 0', 'Class 1']))

# Print custom metrics if needed
for i in range(len(metricas)):
    print(metricas[i])
