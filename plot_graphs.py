import os
import numpy as np
from tqdm import tqdm
import pandas as pd
import torch
import torch.nn as nn
from torchaudio import transforms as T

import graphics as graph
from CNN import Normalize
from Constants import kaggle_path, rambam_path, kauh_path


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


if __name__ == '__main__':
    # lengths_databases()

    visu_lung_sounds(kaggle_path, 'kaggle')
    visu_lung_sounds(rambam_path, 'rambam')
    visu_lung_sounds(kauh_path, 'kauh')
