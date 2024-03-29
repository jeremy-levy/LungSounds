import os
import shutil
import warnings

import numpy as np
import optuna
import pytorch_lightning as pl
import torch
import torch.nn as nn
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers.csv_logs import CSVLogger
from pytorch_lightning.trainer.states import TrainerFn
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, coverage_error, \
    label_ranking_average_precision_score
import joblib

from CNN import CNN14, get_mel_transform
from Constants import get_nb_classes, current_best_p
from DataLoader import DataModule
from utils import save_dict, get_all_metrics


class DNN_clf(pl.LightningModule):
    def __init__(self, learning_rate, kernel_size, pool_type, regularization, single_dataset, leaky,
                 add_augmentation, out_channels, add_standardize, n_fft, n_mels, win_length, hop_length, f_min, f_max,
                 multi_label, class_weight=None):
        super(DNN_clf, self).__init__()
        self.learning_rate = learning_rate
        self.regularization = regularization
        self.nb_classes = get_nb_classes(single_dataset=single_dataset)
        self.multi_label = multi_label
        self.threshold_label = 0.5

        if multi_label is True:
            self.criterion = nn.BCELoss(weight=class_weight)
        else:
            self.criterion = nn.CrossEntropyLoss(weight=class_weight)

        self.train_transform, self.val_transform = get_mel_transform(
            add_augmentation=add_augmentation, add_standardize=add_standardize, n_fft=n_fft, n_mels=n_mels,
            win_length=win_length, hop_length=hop_length, f_min=f_min, f_max=f_max)
        self.clf = CNN14(num_classes=self.nb_classes, do_dropout=True, embed_only=False, out_channels=out_channels,
                         kernel_size=kernel_size, pool_type=pool_type, leaky=leaky)

    def forward(self, x):
        y_pred = self.clf(x)

        if self.multi_label is True:
            y_pred = nn.Sigmoid()(y_pred)
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
        metrics_dict = self.get_metrics(y_hat, y)

        self.log('train_loss', loss, on_step=True, on_epoch=True, logger=True)
        self.log('train_f1_macro', metrics_dict['f1_macro'], on_step=True, on_epoch=True, logger=True)
        self.log('train_acc', metrics_dict['acc'], on_step=True, on_epoch=True, logger=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch

        x = self.val_transform(x)
        y_hat = self(x)

        loss = self.get_loss(y_hat, y)
        metrics_dict = self.get_metrics(y_hat, y)

        self.log('val_loss', loss, on_step=True, on_epoch=True, logger=True)
        self.log('val_f1_macro', metrics_dict['f1_macro'], on_step=True, on_epoch=True, logger=True)
        self.log('val_acc', metrics_dict['acc'], on_step=True, on_epoch=True, logger=True)

        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch

        x = self.val_transform(x)
        y_hat = self(x)

        loss = self.get_loss(y_hat, y)
        metrics_dict = self.get_metrics(y_hat, y, per_class_metrics=True, batch_idx=batch_idx)

        self.log('test_loss', loss, on_step=True, on_epoch=True, logger=True)
        for key in metrics_dict.keys():
            self.log('test_' + key, metrics_dict[key], on_step=True, on_epoch=True, logger=True)

        return loss

    def get_loss(self, y_hat, y):
        if self.multi_label is False:
            _, targets = y.max(dim=1)
            loss = self.criterion(y_hat, targets)
        else:
            loss = self.criterion(y_hat, y)
        return loss

    def get_metrics(self, y_hat, y, per_class_metrics=False, batch_idx=None):
        if self.multi_label is False:
            y_classes = y.cpu().argmax(1)
            y_hat_classes = y_hat.cpu().argmax(1)
        else:
            y_hat_classes = y_hat.cpu().detach().numpy()
            y_classes = y.cpu().detach().numpy()

        if per_class_metrics is True:
            np.save(os.path.join('predictions', 'y_pred_' + str(batch_idx) + '.npy'), y_hat_classes)
            np.save(os.path.join('predictions', 'y_test_' + str(batch_idx) + '.npy'), y_classes)

        if self.multi_label is True:
            y_hat_classes[y_hat_classes < self.threshold_label] = 0
            y_hat_classes[y_hat_classes >= self.threshold_label] = 1

        metrics_dict = get_all_metrics(y_hat_classes, y_classes, multi_label=self.multi_label)
        return metrics_dict


def train(short_sample, p=current_best_p):
    warnings.filterwarnings(action='ignore', category=UndefinedMetricWarning)
    shutil.rmtree('/home/jeremy/ls_clf/predictions')
    os.makedirs('/home/jeremy/ls_clf/predictions', exist_ok=True)
    seed_everything(1)

    if short_sample is True:
        accelerator = 'cpu'
        devices = 1
    else:
        accelerator = 'gpu'
        devices = [2]

    csv_logger = CSVLogger('log', name='lstm', version='0')
    early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=0.00, patience=10, verbose=False, mode="min")
    trainer = Trainer(max_epochs=p['max_epochs'], logger=csv_logger, callbacks=[early_stop_callback],
                      check_val_every_n_epoch=1, devices=devices, accelerator=accelerator, deterministic=True)

    data_module = DataModule(short_sample=short_sample, seq_len=p['seq_len'], batch_size=p['batch_size'],
                             single_dataset=p['single_dataset'], add_sample=p['add_sample'],
                             savgol_filter_add=p['savgol_filter_add'], multi_label=p['multi_label'])
    data_module.setup(stage=TrainerFn.FITTING)
    joblib.dump(data_module.le, '/home/jeremy/ls_clf/predictions/scaler.joblib')

    if (p['multi_label'] is False) and (p['add_class_weight'] is True):
        class_weight = data_module.class_weight
    else:
        class_weight = None

    model = DNN_clf(learning_rate=p['learning_rate'], kernel_size=p['kernel_size'], regularization=p['regularization'],
                    single_dataset=p['single_dataset'], add_augmentation=p['add_augmentation'],
                    class_weight=class_weight, pool_type=p['pool_type'], leaky=p['leaky'],
                    out_channels=p['out_channels'], add_standardize=p['add_standardize'], n_fft=p['n_fft'],
                    n_mels=p['n_mels'], win_length=p['win_length'], hop_length=p['hop_length'], f_min=p['f_min'],
                    f_max=p['f_max'], multi_label=p['multi_label'])

    trainer.fit(model, data_module)
    results = trainer.test(model, datamodule=data_module, ckpt_path='best')[0]
    params = {**p, **results,
              'train_size': data_module.X_train.shape[0], 'test_size': data_module.X_test.shape[0]}

    save_dict(params, os.path.join('data_csv', 'results.csv'))
    return params['test_f1_macro']


def optuna_optimization(short_sample, n_trials):
    if short_sample is True:
        nb_epochs = 2
        batch_size = 4
    else:
        nb_epochs = 100
        batch_size = 16

    def objective(trial):
        p = {
            'seq_len': trial.suggest_int('seq_len', 0.2e6, 0.9e6, step=0.05e6),
            'batch_size': batch_size,
            'max_epochs': nb_epochs,
            'learning_rate': 0.001,
            'kernel_size': trial.suggest_int('kernel_size', 3, 11, step=2),
            'pool_type': trial.suggest_categorical('pool_type', ['avg', 'max', 'avg+max']),
            'regularization': trial.suggest_float('regularization', 0, 1e-4, step=5e-6),
            'single_dataset': False,
            'add_augmentation': trial.suggest_categorical('add_augmentation', [True, False]),
            'add_sample': False,
            'add_class_weight': trial.suggest_categorical('add_class_weight', [True, False]),
            'savgol_filter_add': trial.suggest_categorical('savgol_filter_add', [True, False]),
            'leaky': trial.suggest_categorical('leaky', [True, False]),
            'counter': trial.number,
            'out_channels': 2**trial.suggest_int('out_channels', 2, 6, step=1),
            'add_standardize': trial.suggest_categorical('add_standardize', [True, False]),

            'n_fft': 2**trial.suggest_int('n_fft', 9, 12, step=1),
            'n_mels': 2**trial.suggest_int('n_mels', 4, 6, step=1),
            'win_length': 2**trial.suggest_int('win_length', 9, 12, step=1),
            'f_min': trial.suggest_int('f_min', 0, 100, step=10),
            'f_max': trial.suggest_int('f_max', 1800, 2500, step=100),
            'multi_label': False
        }
        p['hop_length'] = int(p['win_length']/2)

        try:
            f1 = train(short_sample, p)
        except RuntimeError:
        # except KeyboardInterrupt:
            f1 = 0

        torch.cuda.empty_cache()

        shutil.rmtree('/home/jeremy/ls_clf/log/')
        shutil.rmtree('/home/jeremy/ls_clf/lightning_logs/')

        return f1

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials)

    fig = optuna.visualization.plot_param_importances(study)
    fig.write_image('optuna.png')


# TODO: threshold_label adapted
if __name__ == '__main__':
    optuna_optimization(short_sample=False, n_trials=99999)
    # train(short_sample=False)
