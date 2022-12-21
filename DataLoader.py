import h2o
import librosa
import matplotlib.pyplot as plt
import pandas as pd
import os
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Neural Networks
import torch
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader

import pytorch_lightning as pl
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers.csv_logs import CSVLogger


class TimeseriesDataset(Dataset):
    '''
    Custom Dataset subclass.
    Serves as input to DataLoader to transform X
      into sequence data using rolling window.
    DataLoader using this dataset will output batches
      of `(batch_size, seq_len, n_features)` shape.
    Suitable as an input to RNNs.
    '''
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X).float()
        self.y = torch.tensor(y).float()

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, index):
        return (np.expand_dims(self.X[index, :], axis=1), self.y[index])


class DataModule(pl.LightningDataModule):
    def __init__(self, seq_len, batch_size=128, num_workers=0):
        super().__init__()
        self.kaggle_path = '/home/jeremy/dataset_lung_sounds/kaggle/'

        self.seq_len = seq_len
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.le = LabelEncoder()

        self.X_train = None
        self.y_train = None

        self.X_test = None
        self.X_test = None

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        X, y = [], []
        labels_csv = pd.read_csv(os.path.join(self.kaggle_path, 'labels.csv'))

        for i, rec in enumerate(tqdm(os.listdir(os.path.join(self.kaggle_path, 'recordings')))):
            signal, fs = librosa.load(os.path.join(self.kaggle_path, 'recordings', rec))
            id_curr = int(rec.split('_')[0])

            if signal.shape[0] > self.seq_len:
                signal = signal[0: self.seq_len]
            else:
                signal = np.pad(signal.flatten(), (0, int(self.seq_len - len(signal))), constant_values=0)

            X.append(signal)
            y.append(labels_csv.loc[labels_csv.ID == id_curr].label.values[0])

            if i == 40:
                break

        X = np.array(X)
        y = np.array(y)

        self.le.fit(y)
        y = self.le.transform(y)

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        print('X_train', self.X_train.shape)
        print('X_test', self.X_test.shape)

    def train_dataloader(self):
        train_dataset = TimeseriesDataset(self.X_train, self.y_train)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=False,
                                  num_workers=self.num_workers)

        return train_loader

    def test_dataloader(self):
        test_dataset = TimeseriesDataset(self.X_test, self.y_test)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

        return test_loader