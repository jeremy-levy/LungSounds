import os
import numpy as np
from tqdm import tqdm
import pandas as pd
import torch
import torch.nn as nn
from torchaudio import transforms as T
import seaborn as sns
from sklearn.metrics import confusion_matrix
from pytorch_lightning import seed_everything
from pytorch_lightning.trainer.states import TrainerFn

import graphics as graph
from CNN import Normalize
from Constants import kaggle_path, rambam_path, kauh_path, current_best_p
from DataLoader import DataModule


def length_one_database(data_path):
    all_length = []
    for file_name in tqdm(os.listdir(data_path)):
        signal = np.load(os.path.join(data_path, file_name))
        all_length.append(len(signal))

    return all_length


def lengths_databases():
    length_kaggle = length_one_database(kaggle_path + 'recordings_np')
    length_kauh = length_one_database(kauh_path + 'recordings_np')
    length_rambam = length_one_database(rambam_path + 'recordings_np')

    fig, axes = graph.create_figure(subplots=(1, 1), figsize=(8, 8), tight_layout=True)

    bins = np.histogram(np.hstack((length_kaggle, length_kauh, length_rambam)), bins=40)[1]
    axes[0][0].hist(length_kaggle, label='ICBHI', alpha=0.5, color='r', bins=bins, density=True)
    axes[0][0].hist(length_kauh, label='KAUH', alpha=0.5, color='b', bins=bins, density=True)
    axes[0][0].hist(length_rambam, label='Rambam', alpha=0.5, color='g', bins=bins, density=True)

    fontsize = 12

    # axes[0][0].text(-0.12, 1.06, "(a)", fontsize=fontsize, transform=axes[0][0].transAxes)
    # axes[0][1].text(-0.12, 1.06, "(b)", fontsize=fontsize, transform=axes[0][1].transAxes)
    # axes[0][2].text(-0.12, 1.06, "(c)", fontsize=fontsize, transform=axes[0][2].transAxes)

    graph.complete_figure(fig, axes, put_legend=[[True]],
                          xticks_fontsize=fontsize, yticks_fontsize=fontsize,
                          xlabel_fontsize=fontsize, ylabel_fontsize=fontsize,
                          x_titles=[['Length', 'Length', 'Length']],
                          y_titles=[['Number of patients', '', '']],
                          tight_layout=False, savefig=True, main_title='length_database')


def visu_lung_sounds(data_path, data_name):
    fontsize = 12
    labels_csv = pd.read_csv(os.path.join(data_path, 'labels.csv')).rename(columns={'id': 'ID'})
    melspec = T.MelSpectrogram(n_fft=1024, n_mels=64, win_length=1024, hop_length=512, f_min=50, f_max=2000)
    normalize = Normalize()
    melspec = torch.nn.Sequential(melspec, normalize)
    val_transform = nn.Sequential(melspec)

    for i, rec in enumerate(os.listdir(os.path.join(data_path, 'recordings_np'))):
        signal = np.load(os.path.join(data_path, 'recordings_np', rec))
        id_curr = int(rec.split('_')[0])

        label = labels_csv.loc[labels_csv.ID == id_curr].label.values[0]
        mel_stft = val_transform.forward(torch.tensor(signal))

        fig, axes = graph.create_figure(subplots=(1, 1), figsize=(8, 8), tight_layout=True)

        axes[0][0].plot(signal)
        axes[0][0].text(0.5, 0.5, label, fontsize=fontsize, transform=axes[0][0].transAxes)
        
        graph.complete_figure(fig, axes, put_legend=[[False]],
                              xticks_fontsize=fontsize, yticks_fontsize=fontsize,
                              xlabel_fontsize=fontsize, ylabel_fontsize=fontsize,
                              tight_layout=False, savefig=True, main_title=data_name + '_visu_' + str(i))

        if i == 5:
            break


def plot_one_cm(y_pred_class, y_true_class, ax):
    cf_matrix = confusion_matrix(y_true_class, y_pred_class, normalize=None)
    cf_matrix_normalized = confusion_matrix(y_true_class, y_pred_class, normalize='true')

    group_counts = ["{0:0.0f}".format(value) for value in cf_matrix.flatten()]
    group_percentages = ["{0:.0%}".format(value) for value in cf_matrix_normalized.flatten()]

    labels = [f"{v1}\n{v2}" for v1, v2 in zip(group_counts, group_percentages)]
    labels = np.asarray(labels).reshape(int(np.sqrt(len(labels))), int(np.sqrt(len(labels))))

    g = sns.heatmap(cf_matrix_normalized, annot=labels, fmt='', cmap='Blues', ax=ax, cbar=False)
    g.set_xticklabels(g.get_xticklabels(), rotation=45)
    g.set_yticklabels(g.get_yticklabels(), rotation=0)


def plot_cm(scaler, add_str):
    y_pred, y_test = [], []
    for i in range(int(len(os.listdir('/home/jeremy/ls_clf/saved_predictions/exp_1/')) / 2)):
        y_pred += list(np.load('/home/jeremy/ls_clf/saved_predictions/exp_1/y_pred_' + str(i) + '.npy'))
        y_test += list(np.load('/home/jeremy/ls_clf/saved_predictions/exp_1/y_test_' + str(i) + '.npy'))

    display_labels = []
    pathologies = np.unique(y_test)
    for path in pathologies:
        display_labels.append(scaler.categories_[0][path])

    # Plot confusion matrix
    ticks_fontsize, fontsize, letter_fontsize = 15, 15, 15
    fig, axes = graph.create_figure(subplots=(1, 1), figsize=(8, 8))

    y_pred = np.array(y_pred)
    y_pred[y_test == 5] = 5

    plot_one_cm(y_pred, y_test, axes[0][0])

    graph.complete_figure(fig, axes, put_legend=[[False]],
                          xticks_fontsize=ticks_fontsize, yticks_fontsize=ticks_fontsize,
                          xlabel_fontsize=fontsize, ylabel_fontsize=fontsize, tight_layout=True,
                          savefig=True, main_title='confusion_matrix_' + str(add_str),
                          y_titles=[["True label"]], x_titles=[["Predicted label"]],
                          x_ticks_labels=[[display_labels]], y_ticks_labels=[[display_labels]],
                          legend_fontsize=fontsize)


def main_plot_cm():
    seed_everything(1)
    p = current_best_p

    data_module = DataModule(short_sample=False, seq_len=p['seq_len'], batch_size=p['batch_size'],
                             single_dataset=p['single_dataset'], add_sample=p['add_sample'],
                             savgol_filter_add=p['savgol_filter_add'])
    data_module.setup(stage=TrainerFn.FITTING)

    # get_metrics_per_pathology(data_module.le, add_str=p['counter'])
    plot_cm(data_module.le, add_str=p['counter'])


if __name__ == '__main__':
    # lengths_databases()

    # visu_lung_sounds(kaggle_path, 'kaggle')
    # visu_lung_sounds(rambam_path, 'rambam')
    # visu_lung_sounds(kauh_path, 'kauh')

    main_plot_cm()
