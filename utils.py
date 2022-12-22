import os
import pandas as pd
import numpy as np
import librosa
from sklearn.metrics import classification_report
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


def get_metrics_per_pathology(scaler, add_str=''):
    y_pred, y_test = [], []
    for i in range(int(len(os.listdir('/home/jeremy/ls_clf/predictions/')) / 2)):
        y_pred += list(np.load('/home/jeremy/ls_clf/predictions/y_pred_' + str(i) + '.npy'))
        y_test += list(np.load('/home/jeremy/ls_clf/predictions/y_test_' + str(i) + '.npy'))

    results = classification_report(y_test, y_pred, output_dict=True)

    precision_all, recall_all, f1_all, name_path = [], [], [], []
    pathologies = np.unique(y_test)
    for path in pathologies:
        results_path = results[str(path)]

        precision_all.append(results_path['precision'])
        recall_all.append(results_path['recall'])
        f1_all.append(results_path['f1-score'])
        name_path.append(scaler.categories_[0][path])

    res = pd.DataFrame({
        'Pathology': name_path,
        'precision': precision_all,
        'recall': recall_all,
        'f1': f1_all,
    })
    res.to_csv(os.path.join('data_csv', 'results_per_path_' + add_str + '.csv'), index=False)


if __name__ == '__main__':
    convert_wav_to_np(data_path='/home/jeremy/dataset_lung_sounds/kaggle/')
    convert_wav_to_np(data_path='/home/jeremy/dataset_lung_sounds/rambam/')
    convert_wav_to_np(data_path='/home/jeremy/dataset_lung_sounds/KAUH/')
