import os
import numpy as np
from tqdm import tqdm

import graphics as graph
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


if __name__ == '__main__':
    lengths_databases()
