import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def main():
    metrics = pd.read_csv('log/lstm/0/metrics.csv')
    train_loss = metrics[['train_loss_epoch', 'step', 'epoch']][~np.isnan(metrics['train_loss_epoch'])]
    val_loss = metrics[['val_loss_epoch', 'epoch']][~np.isnan(metrics['val_loss_epoch'])]

    fig, axes = plt.subplots(1, 2, figsize=(16, 5), dpi=100)
    axes[0].set_title('Train loss per batch')
    axes[0].plot(train_loss['step'], train_loss['train_loss_epoch'])
    axes[1].set_title('Validation loss per epoch')
    axes[1].plot(val_loss['epoch'], val_loss['val_loss_epoch'], color='orange')
    plt.show(block=True)

    print('MSE:')
    print(f"Train loss: {train_loss['train_loss_epoch'].iloc[-1]:.3f}")
    print(f"Val loss:   {val_loss['val_loss_epoch'].iloc[-1]:.3f}")


main()
