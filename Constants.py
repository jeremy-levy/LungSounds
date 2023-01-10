kaggle_path = '/home/jeremy/dataset_lung_sounds/kaggle/'
rambam_path = '/home/jeremy/dataset_lung_sounds/rambam/'
kauh_path = '/home/jeremy/dataset_lung_sounds/KAUH/'


def get_nb_classes(single_dataset):
    return len(get_labels_keep(single_dataset))


def get_labels_keep(single_dataset):
    # Consider including Bronchiectasis
    if single_dataset is True:
        return ['COPD', 'Healthy', 'Pneumonia', 'URTI']
    return ['COPD', 'Asthma', 'Healthy', 'Pneumonia', 'URTI', 'Heart Failure', 'Lung Fibrosis']


current_best_p = {
    'seq_len': int(0.20e6),
    'batch_size': 16,
    'max_epochs': 100,
    'learning_rate': 0.001,
    'kernel_size': 3,
    'pool_type': 'avg+max',
    'regularization': 0.000045,
    'single_dataset': False,
    'add_augmentation': False,
    'add_sample': False,
    'add_class_weight': True,
    'savgol_filter_add': True,
    'leaky': False,
    'counter': 0,
    'out_channels': 8,
    'add_standardize': True,

    'n_fft': 1024,
    'n_mels': 64,
    'win_length': 512,
    'hop_length': 256,
    'f_min': 80,
    'f_max': 2100,

    'multi_label': True
    }
