import random
import logging
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import librosa
import os


# Only log error messages
tf.get_logger().setLevel(logging.ERROR)
# Set random seed
tf.keras.utils.set_random_seed(42)

# Maximum duration of the input audio file we feed to our Wav2Vec 2.0 model.
MAX_DURATION = 3
# Sampling rate is the number of samples of audio recorded every second
SAMPLING_RATE = 16000
BATCH_SIZE = 32  # Batch-size for training and evaluating our model.
NUM_CLASSES = 2  # Number of classes our dataset will have (11 in our case).
HIDDEN_DIM = 768  # Dimension of our model output (768 in case of Wav2Vec 2.0 - Base).
MAX_SEQ_LENGTH = MAX_DURATION * SAMPLING_RATE  # Maximum length of the input audio file.
# Wav2Vec 2.0 results in an output frequency with a stride of about 20ms.
MAX_FRAMES = 49
MAX_EPOCHS = 2  # Maximum number of training epochs.

MODEL_CHECKPOINT = "facebook/wav2vec2-base"

metadata_path = 'df_filter.csv'
metadata = pd.read_csv(metadata_path)


def load_and_resample_audio(file_path, target_sr=16000):
    # Cargar el archivo de audio usando librosa
    audio, sr = librosa.load(file_path, sr=target_sr)

    # Asegurarse de que el audio esté muestreado a 16kHz
    if sr != target_sr:
        audio = librosa.resample(audio, sr, target_sr)
        sr = target_sr

    return audio, sr


formatted_data = []
audio_folder = '../audios'
# Iterar sobre las filas del DataFrame
for index, row in metadata.iterrows():
    file_path = os.path.join(audio_folder, row['file_name'])  # Obtener la ruta al archivo de audio
    label = row['label']  # Obtener la etiqueta del audio

    # Cargar y muestrear el audio a 16kHz
    audio, sr = load_and_resample_audio(file_path)

    # Crear una entrada en el formato deseado
    entry = {
        'file': file_path,
        'audio': audio,
        'label': label
    }

    # Agregar la entrada al listado de datos formateados
    formatted_data.append(entry)

# Verificar los primeros elementos de los datos formateados
speech_commands_v1  = formatted_data
