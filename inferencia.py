import torch
import librosa
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor 
import soundfile as sf
import tensorflow as tf
import pandas as pd
import numpy as np

import tensorflow as tf

import torch 
import torch.nn as nn
from surgeon_pytorch import Inspect, get_layers

import os
import glob

from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier

import matplotlib.pyplot as plt
import seaborn as sn


#########################--READ_DATASET--#############################################


path_df = "Clasificadores/df_filter.csv"
path_audio = "./audios/"

df = pd.read_csv(path_df)

df["file_cut_path"] = path_audio + df["file_name"] 
df = df.reset_index(drop=True)

##########################--INFERENCE--########################################

def get_features(name):
    def hook(model, input, output):
        features[name] = output.detach()
    return hook

def read_sound(file_name: str):
    if file_name.endswith(".flac"):
        with open(file_name, "rb") as f:
            speech, sample_rate = sf.read(f)
    elif file_name.endswith(".wav"):
        speech, sample_rate = sf.read(file_name)
    else:
        raise NotImplementedError
    assert sample_rate == 16000, "Sound must have a sample rate of 16K"
    speech = tf.constant(speech, dtype=tf.float32)
    return tf.transpose(speech)

#FINETUNED MODEL 
LANG_ID = "en"
MODEL_ID = "jonatasgrosman/wav2vec2-large-english"
SAMPLES = 10

processor = Wav2Vec2Processor.from_pretrained(MODEL_ID)
model = Wav2Vec2ForCTC.from_pretrained(MODEL_ID)

def speech_file_to_array_fn(batch):
    speech_array, sampling_rate = librosa.load(batch["path"], sr=16_000)
    batch["speech"] = speech_array
    batch["sentence"] = batch["sentence"].upper()
    return batch

dataset = df["file_cut_path"]
dataset = dataset.map(read_sound)

inputs = []
for i in range (len(dataset)):
    inputs.append(processor(dataset[i], sampling_rate=16_000, return_tensors="pt", padding=True))


#INTERMEDIATE LAYER AND SAVE TENSORS
#model_intermediate = Inspect(model, layer="wav2vec2.encoder.layers.0.layer_norm")
model_intermediate = Inspect(model, layer="wav2vec2.encoder.layers.14.layer_norm")
log = []

for i in range (len(inputs)):
    y,logits = model_intermediate(inputs[i].input_values.clone().detach(), attention_mask=inputs[i].attention_mask.clone().detach())
    torch.save(logits,os.path.join("./torch_files14" + f"file{i}.pt"))
   
