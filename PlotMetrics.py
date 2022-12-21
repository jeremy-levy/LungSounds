import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def main(metric):
    metrics = pd.read_csv('log/lstm/0/metrics.csv')
    train_loss = metrics[['train_' + metric + '_epoch', 'step',
                          'epoch']][~np.isnan(metrics['train_' + metric + '_epoch'])]
    val_loss = metrics[['val_' + metric + '_epoch', 'epoch']][~np.isnan(metrics['val_' + metric + '_epoch'])]

    fig, axes = plt.subplots(1, 2, figsize=(16, 5), dpi=100)
    axes[0].set_title('Train ' + metric + ' per batch')
    axes[0].plot(train_loss['step'], train_loss['train_' + metric + '_epoch'])
    axes[1].set_title('Validation ' + metric + ' per epoch')
    axes[1].plot(val_loss['epoch'], val_loss['val_' + metric + '_epoch'], color='orange')
    plt.show(block=True)

    print('MSE:')
    print(f"Train: {train_loss['train_' + metric + '_epoch'].iloc[-1]:.3f}")
    print(f"Val:   {val_loss['val_' + metric + '_epoch'].iloc[-1]:.3f}")


main(metric='loss')
main(metric='f1')
main(metric='acc')
