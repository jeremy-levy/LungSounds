from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix
from matplotlib import pyplot as plt
import numpy as np


SNAPSHOTS_DIR = "/home/jeremy/ls_clf/figures/"


# Function which creates a figure according to the number of subplots
def create_figure(**kwargs):
    """ This function is responsible to build figures based on the different arguments the user provides, among which:
        - figsize
        - subplots
        - sharex (Link between the different x axis over the subplots)
        - sharey (Link between the different y axis over the subplots)
    """
    fig = plt.figure(figsize=kwargs.get('figsize', (20, 10)))
    subplots = kwargs.get('subplots', (1, 1))
    sharex = kwargs.get('sharex', False)
    sharey = kwargs.get('sharey', False)
    axes = np.empty(subplots, dtype=object)
    ax_num = 1
    for i in range(axes.shape[0]):
        for j in range(axes.shape[1]):
            if ax_num == 1:
                axes[i][j] = plt.subplot(subplots[0], subplots[1], ax_num)
            elif sharex and sharey:
                axes[i][j] = plt.subplot(subplots[0], subplots[1], ax_num, sharex=axes[0][0], sharey=axes[0][0])
            elif sharex:
                axes[i][j] = plt.subplot(subplots[0], subplots[1], ax_num, sharex=axes[0][0])
            elif sharey:
                axes[i][j] = plt.subplot(subplots[0], subplots[1], ax_num, sharey=axes[0][0])
            else:
                axes[i][j] = plt.subplot(subplots[0], subplots[1], ax_num)
            axes[i][j].spines['right'].set_visible(False)
            axes[i][j].spines['top'].set_visible(False)
            ax_num += 1

    if kwargs.get('tight_layout', False):
        fig.tight_layout()

    return fig, axes


# Function which completes a figure with the different titles.
# Should be called after the creation of the figure and plotting the data
def complete_figure(fig, axes, **kwargs):

    """ This function is responsible to complete figures based on the different arguments the user provides, among which:
        - different fontiszes
        - display legends
        - limits of x and y axes
        - x and y ticks
        - save figure (default directory is cts.SNAPSHOTS_DIR)
        These parameters should be provided in a 2D array where the dimensions are: (n_horizontal_subplots, n_vertical_subplots)
    """
    without_xticks_fontsize = kwargs.get('without_xticks_fontsize', False)
    xticks_fontsize = kwargs.get('xticks_fontsize', 28)
    yticks_fontsize = kwargs.get('yticks_fontsize', 28)
    xlabel_fontsize = kwargs.get('xlabel_fontsize', 28)
    ylabel_fontsize = kwargs.get('ylabel_fontsize', 28)
    frameon = kwargs.get('frameon', False)
    x_titles = kwargs.get('x_titles', '' * np.ones(axes.shape, dtype=object))  # No titles
    y_titles = kwargs.get('y_titles', '' * np.ones(axes.shape, dtype=object))  # No titles
    x_lim = kwargs.get('x_lim', 'auto' * np.ones(axes.shape, dtype=object))
    y_lim = kwargs.get('y_lim', 'auto' * np.ones(axes.shape, dtype=object))
    x_ticks = kwargs.get('x_ticks', 'auto' * np.ones(axes.shape, dtype=object))
    y_ticks = kwargs.get('y_ticks', 'auto' * np.ones(axes.shape, dtype=object))
    x_ticks_labels = kwargs.get('x_ticks_labels', 'auto' * np.ones(axes.shape, dtype=object))
    y_ticks_labels = kwargs.get('y_ticks_labels', 'auto' * np.ones(axes.shape, dtype=object))
    put_legend = kwargs.get('put_legend', False * np.ones(axes.shape, dtype=bool))
    loc_legend = kwargs.get('loc_legend', 'best' * np.ones(axes.shape, dtype=object))
    rotation_xticks = kwargs.get('rotation_xticks', 'auto' * np.ones(axes.shape, dtype=object))

    for i in range(axes.shape[0]):
        for j in range(axes.shape[1]):
            axes[i][j].set_xlabel(x_titles[i][j], fontsize=xlabel_fontsize)
            axes[i][j].set_ylabel(y_titles[i][j], fontsize=ylabel_fontsize)
            if without_xticks_fontsize is False:
                axes[i][j].tick_params(axis='x', labelsize=xticks_fontsize)
            axes[i][j].tick_params(axis='y', labelsize=yticks_fontsize)
            if put_legend[i][j]:
                axes[i][j].legend(fontsize=kwargs.get('legend_fontsize', 28), loc=loc_legend[i][j], frameon=frameon)
            if x_lim[i][j] != 'auto':
                axes[i][j].set_xlim(x_lim[i][j])
            if y_lim[i][j] != 'auto':
                axes[i][j].set_ylim(y_lim[i][j])
            if x_ticks[i][j] != 'auto':
                axes[i][j].set_xticks(x_ticks[i][j])
            if y_ticks[i][j] != 'auto':
                axes[i][j].set_yticks(y_ticks[i][j])
            if x_ticks_labels[i][j] != 'auto':
                if rotation_xticks[i][j] != 'auto':
                    axes[i][j].set_xticklabels(x_ticks_labels[i][j], rotation=rotation_xticks[i][j])
                else:
                    axes[i][j].set_xticklabels(x_ticks_labels[i][j])
            if y_ticks_labels[i][j] != 'auto':
                axes[i][j].set_yticklabels(y_ticks_labels[i][j])

    plt.suptitle(kwargs.get('suptitle', ''), fontsize=kwargs.get('suptitle_fontsize', 28))
    if kwargs.get('tight_layout', True) is True:
        fig.tight_layout()

    if kwargs.get('savefig', False):
        plt.savefig(SNAPSHOTS_DIR + (kwargs.get('main_title', 'NoName') + '.png'),
                    bbox_inches='tight')

        if kwargs.get('pdf', False):
            plt.savefig(SNAPSHOTS_DIR + (kwargs.get('main_title', 'NoName') + '.pdf'),
                        bbox_inches='tight')

    plt.close(fig)


def model_metrics(data, label, predicted, beta=1):
    """ This function returns different statistical binary metrics based on the data (output score/probabilities),
        the predicted and the actual labels. Function established for binary classification only.
    :param data:                The output score/probabilities of the algorithm.
    :param label:               The actual labels of the examples.
    :param predicted:           The predicted labels of the examples.
    :param beta:                Index for the F-beta measure.
    :returns fbeta:             The F-beta measure (https://en.wikipedia.org/wiki/F1_score)
    :returns AUROC:             The Area Under the ROC Curve. (https://glassboxmedicine.com/2019/02/23/measuring-performance-auc-auroc/)
    :returns sensitivity:       The Sensitivity (or Recall) of the algorithm. (https://en.wikipedia.org/wiki/Sensitivity_and_specificity)
    :returns specificity:       The Specificity (or False Positive Rate) of the algorithm. (https://en.wikipedia.org/wiki/Sensitivity_and_specificity)
    :returns PPV:               The Positive Predictive Value (or Precision) of the algorithm. (https://en.wikipedia.org/wiki/Precision_and_recall)
    """
    AUROC = roc_auc_score(label, data)
    accuracy = accuracy_score(label, predicted)
    TN, FP, FN, TP = confusion_matrix(label, predicted).ravel()
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    sensitivity = recall
    specificity = TN / (TN + FP)
    PPV = precision
    NPV = TN / (TN + FN)
    fbeta = (1 + beta ** 2) * precision * recall / ((beta ** 2) * precision + recall)
    print("Accuracy: " + str(accuracy))
    print("F" + str(beta) + "-Score: " + str(fbeta))
    print("Sensitivity: " + str(sensitivity))
    print("Specificity: " + str(specificity))
    print("PPV: " + str(PPV))
    print("NPV: " + str(NPV))
    print("AUROC: " + str(AUROC))
    print(confusion_matrix(label, predicted))
    return fbeta, AUROC, sensitivity, specificity, PPV
