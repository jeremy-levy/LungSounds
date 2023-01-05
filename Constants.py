kaggle_path = '/home/jeremy/dataset_lung_sounds/kaggle/'
rambam_path = '/home/jeremy/dataset_lung_sounds/rambam/'
kauh_path = '/home/jeremy/dataset_lung_sounds/KAUH/'


def get_nb_classes(single_dataset):
    return len(get_labels_keep(single_dataset))


def get_labels_keep(single_dataset):
    if single_dataset is True:
        return ['COPD', 'Healthy', 'Pneumonia', 'URTI']
    return ['COPD', 'Asthma', 'Healthy', 'Pneumonia', 'URTI', 'Heart Failure']


current_best_p = {
        'seq_len': int(0.45e6),
        'batch_size': 16,
        'max_epochs': 100,
        'learning_rate': 0.001,
        'kernel_size': 3,
        'pool_type': 'max',
        'regularization': 0,
        'single_dataset': False,
        'add_augmentation': True,
        'add_sample': False,
        'add_class_weight': True,
        'savgol_filter_add': True,
        'leaky': True,
        'counter': 0,
        'out_channels': 4,
        'add_standardize': True,
    }