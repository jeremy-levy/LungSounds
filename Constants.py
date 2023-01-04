kaggle_path = '/home/jeremy/dataset_lung_sounds/kaggle/'
rambam_path = '/home/jeremy/dataset_lung_sounds/rambam/'
kauh_path = '/home/jeremy/dataset_lung_sounds/KAUH/'


def get_nb_classes(single_dataset):
    return len(get_labels_keep(single_dataset))


def get_labels_keep(single_dataset):
    if single_dataset is True:
        return ['COPD', 'Healthy', 'Pneumonia', 'URTI']
    return ['COPD', 'Asthma', 'Healthy', 'Pneumonia', 'URTI', 'Heart Failure']
