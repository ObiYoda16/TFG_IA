import os
import numpy as np
import pandas as pd
import librosa
import parselmouth
from parselmouth.praat import call
from scipy.fftpack import dct

# Directorio de audios
audio_dir = '../audios/'


# Función para extraer MFCC
def extract_mfcc(y, sr, n_mfcc=13):
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    mfccs = np.mean(mfccs.T, axis=0)
    return mfccs


# Función para extraer GTCC
def extract_gtcc(y, sr, n_gtcc=13):
    S = np.abs(librosa.stft(y)) ** 2
    mel_basis = librosa.filters.mel(sr=sr, n_fft=2048, n_mels=n_gtcc)
    mel_S = np.dot(mel_basis, S)
    log_mel_S = librosa.power_to_db(mel_S)
    gtccs = dct(log_mel_S, type=2, axis=0, norm='ortho')[1:(n_gtcc + 1)]
    gtccs = np.mean(gtccs.T, axis=0)
    return gtccs


# Función para extraer formantes usando Praat
def extract_formants(y, sr):
    sound = parselmouth.Sound(y, sr)
    formants = call(sound, "To Formant (burg)", 0.02, 5, 5500, 0.025, 50)
    f1 = call(formants, "Get mean", 1, 0, 0, "Hertz")
    f2 = call(formants, "Get mean", 2, 0, 0, "Hertz")
    f3 = call(formants, "Get mean", 3, 0, 0, "Hertz")
    return [f1, f2, f3]


# Función para extraer pitch usando Praat
def extract_pitch(y, sr):
    try:
        sound = parselmouth.Sound(y, sr)
        pitch = call(sound, "To Pitch", 0.0, 75, 600)
        mean_pitch = call(pitch, "Get mean", 0, 0, "Hertz")
        if np.isnan(mean_pitch):
            mean_pitch = 0
        return mean_pitch
    except parselmouth.PraatError:
        return 0


# Función para extraer Jitter, Shimmer y Fundamental Frequency usando Praat
def extract_jitter_shimmer(y, sr):
    try:
        sound = parselmouth.Sound(y, sr)
        point_process = call(sound, "To PointProcess (periodic, cc)", 75, 600)
        jitter_local = call(point_process, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
        jitter_local_absolute = call(point_process, "Get jitter (local, absolute)", 0, 0, 0.0001, 0.02, 1.3)
        jitter_rap = call(point_process, "Get jitter (rap)", 0, 0, 0.0001, 0.02, 1.3)
        jitter_ppq5 = call(point_process, "Get jitter (ppq5)", 0, 0, 0.0001, 0.02, 1.3)
        jitter_ddp = call(point_process, "Get jitter (ddp)", 0, 0, 0.0001, 0.02, 1.3)
        shimmer_local = call([sound, point_process], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
        shimmer_local_dB = call([sound, point_process], "Get shimmer (local_dB)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
        shimmer_apq3 = call([sound, point_process], "Get shimmer (apq3)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
        shimmer_apq5 = call([sound, point_process], "Get shimmer (apq5)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
        shimmer_apq11 = call([sound, point_process], "Get shimmer (apq11)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
        fundamental_frequency = call(sound, "Get mean", 0, 0, "Hertz")

        # Verificación de valores NaN
        jitters = [jitter_local, jitter_local_absolute, jitter_rap, jitter_ppq5, jitter_ddp]
        shimmers = [shimmer_local, shimmer_local_dB, shimmer_apq3, shimmer_apq5, shimmer_apq11]
        jitters = [0 if np.isnan(j) else j for j in jitters]
        shimmers = [0 if np.isnan(s) else s for s in shimmers]
        if np.isnan(fundamental_frequency):
            fundamental_frequency = 0

        return jitters, shimmers, fundamental_frequency
    except parselmouth.PraatError:
        return [0] * 5, [0] * 5, 0


# Lista para almacenar los resultados
results = []

# Recorrer todos los archivos de audio en el directorio
df = pd.read_csv('df_filter.csv')
for index, row in df.iterrows():
    file_path = os.path.join(audio_dir, row['file_name'])

    # Cargar el audio
    y, sr = librosa.load(file_path, sr=16000)

    # Extraer características
    mfccs = extract_mfcc(y, sr)
    gtccs = extract_gtcc(y, sr)
    formants = extract_formants(y, sr)
    pitch = extract_pitch(y, sr)
    jitters, shimmers, fundamental_frequency = extract_jitter_shimmer(y, sr)

    # Almacenar resultados en un diccionario
    features = {
        'filename': row['file_name'],
        'pitch': pitch,
        'f1': formants[0],
        'f2': formants[1],
        'f3': formants[2],
        'fundamental_frequency': fundamental_frequency,
        'label': row['label']
    }

    for i, jitter in enumerate(jitters, 1):
        features[f'jitter_{i}'] = jitter

    for i, shimmer in enumerate(shimmers, 1):
        features[f'shimmer_{i}'] = shimmer

    for i in range(len(mfccs)):
        features[f'mfcc_{i + 1}'] = mfccs[i]

    for i in range(len(gtccs)):
        features[f'gtcc_{i + 1}'] = gtccs[i]

    results.append(features)

# Convertir resultados a un DataFrame de pandas
df = pd.DataFrame(results)

# Guardar los resultados en un archivo CSV
df.to_csv('audio_features.csv', index=False)
