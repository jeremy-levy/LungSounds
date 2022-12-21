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

from DataLoader import DataModule

CLASSES = 1


class DNN_clf(pl.LightningModule):
    def __init__(self, num_layers, batch_size, learning_rate, criterion):
        super(DNN_clf, self).__init__()
        self.batch_size = batch_size
        self.criterion = criterion
        self.learning_rate = learning_rate

        all_layers = []
        in_channels = 1
        out_channels = 2
        for i in range(num_layers):
            all_layers.append(nn.Conv1d(in_channels, out_channels, kernel_size=11))
            all_layers.append(nn.ReLU())
            all_layers.append(nn.BatchNorm2d(out_channels))
            all_layers.append(nn.MaxPool1d(5))

            in_channels = out_channels
            out_channels = in_channels * 2
        self.cnn_part = nn.Sequential(*all_layers)

        clf = [nn.Flatten(), nn.Linear(1152, CLASSES), nn.Softmax()]
        self.clf = nn.Sequential(*clf)

    def forward(self, x):
        cnn_feat = self.cnn_part(x)

        y_pred = self.clf(cnn_feat)
        return y_pred

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        result = pl.TrainResult(loss)
        result.log('train_loss', loss)
        return result

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        result = pl.EvalResult(checkpoint_on=loss)
        result.log('val_loss', loss)
        return result

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        result = pl.EvalResult()
        result.log('test_loss', loss)
        return result


def main():
    p = dict(
        seq_len=int(0.75e6),
        batch_size=16,
        criterion=nn.MSELoss(),
        max_epochs=10,
        n_features=1,
        hidden_size=100,
        num_layers=1,
        dropout=0.2,
        learning_rate=0.001,
    )

    seed_everything(1)

    csv_logger = CSVLogger('', name='lstm', version='0')

    trainer = Trainer(max_epochs=p['max_epochs'], logger=csv_logger)

    model = DNN_clf(
        batch_size=p['batch_size'],
        criterion=p['criterion'],
        num_layers=p['num_layers'],
        learning_rate=p['learning_rate']
    )

    data_module = DataModule(seq_len=p['seq_len'])

    trainer.fit(model, data_module)
    trainer.test(model, datamodule=data_module)


main()
