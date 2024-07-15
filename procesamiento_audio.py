# audio_processing.py

import pandas as pd
import torch
from sklearn.model_selection import train_test_split
import numpy as np

class AudioProcessor:
    def __init__(self, path_df, path_audio):
        self.path_df = "df_filter.csv"
        self.path_audio = path_audio
        self.load_data()

    def load_data(self):
        df = pd.read_csv(self.path_df)
        df["file_cut_path"] = self.path_audio + df["file_name"]
        df = df.reset_index(drop=True)
        self.dataset = df["file_cut_path"]

        self.log = []
        for i in range(len(df)):
            self.log.append(torch.load("../torch_files14/" + f"torch_files14file{i}.pt"))

        self.y = df["label"]
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.log, self.y, random_state=42, stratify=self.y
        )

        self.process_tensors()

    def process_tensors(self):
        for i in range(len(self.X_train)):
            self.X_train[i] = self.X_train[i][0].detach().numpy()
            self.X_train[i] = np.concatenate(self.X_train[i])

        for i in range(len(self.X_test)):
            self.X_test[i] = self.X_test[i][0].detach().numpy()
            self.X_test[i] = np.concatenate(self.X_test[i])
