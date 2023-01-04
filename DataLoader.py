import os

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import scipy
# Neural Networks
import torch
from pytorch_lightning.trainer.states import TrainerFn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from tqdm import tqdm
from scipy.signal import savgol_filter

from Constants import kaggle_path, rambam_path, kauh_path, get_labels_keep
from utils import get_class_weight


class TimeseriesDataset(Dataset):
    """
    Custom Dataset subclass.
    Serves as input to DataLoader to transform X
      into sequence data using rolling window.
    DataLoader using this dataset will output batches
      of `(batch_size, seq_len, n_features)` shape.
    """
    def __init__(self, X: np.ndarray, y: scipy.sparse.csr.csr_matrix):
        self.X = torch.tensor(X).float()
        self.y = torch.tensor(y.toarray()).float()

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, index):
        return np.expand_dims(self.X[index, :], axis=0), self.y[index]


class DataModule(pl.LightningDataModule):
    def __init__(self, short_sample, seq_len, batch_size, savgol_filter_add, num_workers=5, single_dataset=False, add_sample=False):
        super().__init__()
        self.short_sample = short_sample
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.single_dataset = single_dataset
        self.labels_keep = get_labels_keep(single_dataset)
        self.add_sample = add_sample
        self.savgol_filter_add = savgol_filter_add

        self.le = OneHotEncoder()

        self.X_train, self.y_train = None, None
        self.X_val, self.y_val = None, None
        self.X_test, self.y_test = None, None
        self.class_weight = None

    def prepare_data(self):
        pass

    def parse_one_database(self, data_path, desc):
        X, y = [], []
        labels_csv = pd.read_csv(os.path.join(data_path, 'labels.csv')).rename(columns={'id': 'ID'})
        unkeep_labels = []

        for i, rec in enumerate(tqdm(os.listdir(os.path.join(data_path, 'recordings_np')), desc=desc)):
            # signal, fs = librosa.load(os.path.join(data_path, 'recordings', rec))
            signal = np.load(os.path.join(data_path, 'recordings_np', rec))
            if self.savgol_filter_add is True:
                signal = savgol_filter(signal, window_length=11, polyorder=3)

            id_curr = int(rec.split('_')[0])

            if signal.shape[0] > self.seq_len:
                signal = signal[0: self.seq_len]
            else:
                signal = np.pad(signal.flatten(), (0, int(self.seq_len - len(signal))), constant_values=0)

            label = labels_csv.loc[labels_csv.ID == id_curr].label.values[0]
            if label in self.labels_keep:
                X.append(signal)
                y.append(label)
            else:
                unkeep_labels.append(label)

            if (self.short_sample is True) and (i >= 30):
                break

        X = np.array(X)
        y = np.array(y).reshape(-1, 1)

        print('labels not used:', np.unique(unkeep_labels))
        return X, y

    def setup(self, stage=None):
        if stage == TrainerFn.FITTING:
            if self.single_dataset is False:
                X_kaggle, y_kaggle = self.parse_one_database(kaggle_path, desc='ICBHI')
                X_rambam, y_rambam = self.parse_one_database(rambam_path, desc='Rambam')
                X_kauh, y_kauh = self.parse_one_database(kauh_path, desc='KAUH')

                X = np.concatenate((X_kaggle, X_rambam, X_kauh))
                y = np.concatenate((y_kaggle, y_rambam, y_kauh))
            else:
                X, y = self.parse_one_database(kaggle_path, desc='ICBHI')

            self.le.fit(y)
            y = self.le.transform(y)

            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2,
                                                                                    random_state=32)
            self.X_val = self.X_test
            self.y_val = self.y_test

            self.class_weight = get_class_weight(self.y_train)

    def train_dataloader(self):
        if self.add_sample is True:
            sampler = WeightedRandomSampler(self.class_weight, len(self.class_weight))
        else:
            sampler = None

        train_dataset = TimeseriesDataset(self.X_train, self.y_train)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=False,
                                  num_workers=self.num_workers, sampler=sampler)

        return train_loader

    def val_dataloader(self):
        train_dataset = TimeseriesDataset(self.X_val, self.y_val)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=False,
                                  num_workers=self.num_workers)

        return train_loader

    def test_dataloader(self):
        test_dataset = TimeseriesDataset(self.X_test, self.y_test)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=0)

        return test_loader

    def predict_dataloader(self):
        return self.test_dataloader()


if __name__ == '__main__':
    datamodule = DataModule(short_sample=False, seq_len=int(0.75e6), batch_size=16, single_dataset=False,
                            savgol_filter_add=True)
    datamodule.setup(stage=TrainerFn.FITTING)

    # y_train = np.argmax(datamodule.y_train.toarray(), axis=1)
    # unique, counts = np.unique(y_train, return_counts=True)
    # print(dict(zip(unique, counts)))

    # y_test = np.argmax(datamodule.y_test.toarray(), axis=1)
    # unique, counts = np.unique(y_test, return_counts=True)
    # print(dict(zip(unique, counts)))
