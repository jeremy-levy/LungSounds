import os
import pandas as pd
import numpy as np
import librosa
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm
import torch
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, coverage_error, \
    label_ranking_average_precision_score


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


def get_all_metrics(y_hat_classes, y_classes, multi_label):
    f1_weighted = f1_score(y_hat_classes, y_classes, average='weighted')
    f1_macro = f1_score(y_hat_classes, y_classes, average='macro')
    acc = accuracy_score(y_hat_classes, y_classes)
    precision = precision_score(y_hat_classes, y_classes, average='weighted')
    recall = recall_score(y_hat_classes, y_classes, average='weighted')
    accuracy = accuracy_score(y_hat_classes, y_classes)

    metrics_dict = dict(f1_weighted=f1_weighted, f1_macro=f1_macro, acc=acc, precision=precision, recall=recall,
                        accuracy=accuracy)

    if multi_label is True:
        metrics_dict['cov_error'] = coverage_error(y_classes, y_hat_classes)
        metrics_dict['label_ranking_average_precision_score'] = label_ranking_average_precision_score(y_classes,
                                                                                                      y_hat_classes)

    return metrics_dict


def get_metrics_per_pathology(scaler, y_pred, y_test, multilabel=False):
    results = classification_report(y_test, y_pred, output_dict=True)

    precision_all, recall_all, f1_all, support_all, name_path = [], [], [], [], []
    pathologies = np.unique(y_test)
    for path in pathologies:
        results_path = results[str(int(path))]

        precision_all.append(results_path['precision'])
        recall_all.append(results_path['recall'])
        f1_all.append(results_path['f1-score'])
        support_all.append(results_path['support'])
        name_path.append(scaler.categories_[0][int(path)])

    res = pd.DataFrame({
        'Pathology': name_path,
        'precision': precision_all,
        'recall': recall_all,
        'f1': f1_all,
        'support': support_all,
    })

    if multilabel is False:
        matrix = confusion_matrix(y_test, y_pred)
        acc = matrix.diagonal() / matrix.sum(axis=1)
        res['accuracy'] = acc
    else:
        acc = []
        for i in range(y_pred.shape[1]):
            acc.append(accuracy_score(y_pred[:, i], y_test[:, i]))
        res['accuracy'] = acc

    print(res)
    res.to_csv(os.path.join('data_csv', 'results_per_path.csv'), index=False)

    return res


def get_metrics_per_pathology_multilabel(scaler, y_pred, y_true):
    precision_all, recall_all, f1_all, support_all, acc_all, name_path = [], [], [], [], [], []
    for i in range(y_true.shape[1]):
        name_path.append(scaler.classes_[i])

        y_true_curr = y_true[:, i]
        y_pred_curr = y_pred[:, i]

        precision_all.append(precision_score(y_true_curr, y_pred_curr))
        recall_all.append(recall_score(y_true_curr, y_pred_curr))
        f1_all.append(f1_score(y_true_curr, y_pred_curr))
        support_all.append(int(np.sum(y_true_curr)))
        acc_all.append(accuracy_score(y_pred[:, i], y_true[:, i]))

    res = pd.DataFrame({
        'Pathology': name_path,
        'precision': precision_all,
        'recall': recall_all,
        'f1': f1_all,
        'support': support_all,
        'accuracy': acc_all,
    })
    return res


def get_class_weight(y_train):
    y_train = np.argmax(y_train.toarray(), axis=1)
    class_weight = []
    labels = np.unique(y_train)
    for i in range(len(labels)):
        num_label = y_train[y_train == i].shape[0]
        weight_curr = 1 - (num_label / y_train.shape[0])
        class_weight.append(weight_curr)

    class_weight = torch.FloatTensor(class_weight)
    return class_weight


if __name__ == '__main__':
    convert_wav_to_np(data_path='/home/jeremy/dataset_lung_sounds/kaggle/')
    convert_wav_to_np(data_path='/home/jeremy/dataset_lung_sounds/rambam/')
    convert_wav_to_np(data_path='/home/jeremy/dataset_lung_sounds/KAUH/')
