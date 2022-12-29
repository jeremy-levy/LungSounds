import os
import warnings
from sklearn.exceptions import UndefinedMetricWarning
import numpy as np

import pytorch_lightning as pl
import torch
import torch.nn as nn
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers.csv_logs import CSVLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from sklearn.metrics import classification_report
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score

from CNN import CNN14, get_mel_transform
from Classic_CNN import get_cnn
from Constants import NB_CLASSES
from DataLoader import DataModule
from utils import WrongParameter, save_dict, get_metrics_per_pathology


class DNN_clf(pl.LightningModule):
    def __init__(self, num_layers, learning_rate, criterion, kernel_size, seq_len, dropout, hidden_size, model, device,
                 regularization):
        super(DNN_clf, self).__init__()
        self.criterion = criterion
        self.learning_rate = learning_rate
        self.model = model
        self.device_ = device
        self.regularization = regularization

        if model == 'cnn_1d':
            self.cnn_part, self.maxpool_cnn, self.clf = get_cnn(num_layers, kernel_size, dropout, seq_len, hidden_size)
        elif model == 'cnn_2d':
            self.train_transform, self.val_transform = get_mel_transform(add_augmentation=False)
            self.clf = CNN14(num_classes=NB_CLASSES, do_dropout=False, embed_only=False, device=self.device_)
        else:
            raise WrongParameter('model must be in {cnn_1d, cnn_2d}')

    def forward(self, x):
        if self.model == 'cnn_1d':
            cnn_feat = self.cnn_part(x)
            cnn_feat = cnn_feat.permute(0, 2, 1)
            cnn_feat = self.maxpool_cnn(cnn_feat)

            y_pred = self.clf(cnn_feat)
        elif self.model == 'cnn_2d':
            y_pred = self.clf(x)
        else:
            raise WrongParameter('model must be in {cnn_1d, cnn_2d}')

        return y_pred

    def configure_optimizers(self):
        if self.regularization == 0:
            return torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=self.regularization)

    def training_step(self, batch, batch_idx):
        x, y = batch

        x = self.train_transform(x)
        y_hat = self(x)

        loss = self.get_loss(y_hat, y)
        f1, acc, precision, recall = self.get_metrics(y_hat, y)

        self.log('train_loss', loss, on_step=True, on_epoch=True, logger=True)
        self.log('train_f1', f1, on_step=True, on_epoch=True, logger=True)
        self.log('train_acc', acc, on_step=True, on_epoch=True, logger=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch

        x = self.val_transform(x)
        y_hat = self(x)

        loss = self.get_loss(y_hat, y)
        f1, acc, precision, recall = self.get_metrics(y_hat, y)

        self.log('val_loss', loss, on_step=True, on_epoch=True, logger=True)
        self.log('val_f1', f1, on_step=True, on_epoch=True, logger=True)
        self.log('val_acc', acc, on_step=True, on_epoch=True, logger=True)

        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch

        x = self.val_transform(x)
        y_hat = self(x)

        loss = self.get_loss(y_hat, y)
        f1, acc, precision, recall = self.get_metrics(y_hat, y, per_class_metrics=True, batch_idx=batch_idx)

        self.log('test_loss', loss, on_step=True, on_epoch=True, logger=True)
        self.log('test_f1', f1, on_step=True, on_epoch=True, logger=True)
        self.log('test_acc', acc, on_step=True, on_epoch=True, logger=True)
        self.log('test_precision', precision, on_step=True, on_epoch=True, logger=True)
        self.log('test_recall', recall, on_step=True, on_epoch=True, logger=True)

        return loss

    def get_loss(self, y_hat, y):
        _, targets = y.max(dim=1)
        loss = self.criterion(y_hat, targets)
        return loss

    @staticmethod
    def get_metrics(y_hat, y, per_class_metrics=False, batch_idx=None):
        y_classes = y.cpu().argmax(1)
        y_hat_classes = y_hat.cpu().argmax(1)

        f1 = f1_score(y_hat_classes, y_classes, average='weighted')
        acc = accuracy_score(y_hat_classes, y_classes)
        precision = precision_score(y_hat_classes, y_classes, average='weighted')
        recall = recall_score(y_hat_classes, y_classes, average='weighted')

        if per_class_metrics is True:
            np.save(os.path.join('predictions', 'y_pred_' + str(batch_idx) + '.npy'), y_hat_classes)
            np.save(os.path.join('predictions', 'y_test_' + str(batch_idx) + '.npy'), y_classes)

        return f1, acc, precision, recall


def main(short_sample, regularization):
    warnings.filterwarnings(action='ignore', category=UndefinedMetricWarning)

    if short_sample is True:
        nb_epochs = 2
        batch_size = 4
    else:
        nb_epochs = 100
        batch_size = 16

    p = dict(
        seq_len=int(0.35e6),
        batch_size=batch_size,
        criterion=nn.CrossEntropyLoss(),
        max_epochs=nb_epochs,
        num_layers=5,
        dropout=0.2,
        learning_rate=0.001,
        kernel_size=7,
        hidden_size=16,
        model='cnn_2d',      # cnn_2d / cnn_1d
        regularization=regularization
    )

    seed_everything(1)
    csv_logger = CSVLogger('log', name='lstm', version='0')
    early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=0.00, patience=10, verbose=False, mode="min")
    trainer = Trainer(max_epochs=p['max_epochs'], logger=csv_logger, callbacks=[early_stop_callback],
                      check_val_every_n_epoch=1, devices=[2], accelerator='gpu')

    model = DNN_clf(criterion=p['criterion'], num_layers=p['num_layers'], learning_rate=p['learning_rate'],
                    seq_len=p['seq_len'], kernel_size=p['kernel_size'], dropout=p['dropout'],
                    hidden_size=p['hidden_size'], model=p['model'], device='cuda:2',
                    regularization=p['regularization'])

    data_module = DataModule(short_sample=short_sample, seq_len=p['seq_len'], batch_size=p['batch_size'])

    trainer.fit(model, data_module)
    results = trainer.test(model, datamodule=data_module, ckpt_path='best')[0]
    get_metrics_per_pathology(data_module.le)

    params = {**p, **results}
    save_dict(params, os.path.join('data_csv', 'results.csv'))


# TODO: Pre-processing, add multi-label samples
if __name__ == '__main__':
    regularization_it = [0, 1e-7, 1e-6, 1e-5, 1e-4]

    for regularization in regularization_it:
        main(short_sample=False, regularization=regularization)
        torch.cuda.empty_cache()
