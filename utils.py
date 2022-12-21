import os
import pandas as pd
import numpy as np
import librosa
from tqdm import tqdm


class WrongParameter(Exception):
    pass


def save_dict(params_it, file_name, override=False):
    scores_it = pd.DataFrame(params_it.copy(), index=[0])

    if override is True:
        scores_it.to_csv(file_name)
    else:
        if os.path.exists(file_name):
            scores_it.to_csv(file_name, mode='a', header=False)
        else:
            scores_it.to_csv(file_name)


def convert_wav_to_np(data_path):
    os.makedirs(os.path.join(data_path, 'recordings_np'), exist_ok=True)
    for file_name in tqdm(os.listdir(os.path.join(data_path, 'recordings'))):
        signal, fs = librosa.load(os.path.join(data_path, 'recordings', file_name))
        np.save(os.path.join(data_path, 'recordings_np', file_name[0:-3] + 'npy'), signal)


convert_wav_to_np(data_path='/home/jeremy/dataset_lung_sounds/kaggle/')
convert_wav_to_np(data_path='/home/jeremy/dataset_lung_sounds/rambam/')
convert_wav_to_np(data_path='/home/jeremy/dataset_lung_sounds/KAUH/')
