import torch.nn as nn

from Constants import NB_CLASSES


def get_cnn(num_layers, kernel_size, dropout, seq_len, hidden_size):
    all_layers = []
    in_channels = 1
    out_channels = 2
    for i in range(num_layers):
        all_layers.append(nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size,
                                    padding=int((kernel_size - 1) / 2)))
        all_layers.append(nn.LeakyReLU())
        all_layers.append(nn.BatchNorm1d(out_channels))
        all_layers.append(nn.MaxPool1d(2))
        all_layers.append(nn.Dropout(dropout))

        in_channels = out_channels
        out_channels = in_channels * 2

    cnn_part = nn.Sequential(*all_layers)
    maxpool_cnn = nn.MaxPool1d(in_channels)

    input_size_clf = int(seq_len / (2 ** num_layers))

    clf = [nn.Flatten(),

           # First block
           nn.Linear(input_size_clf, hidden_size), nn.BatchNorm1d(hidden_size), nn.LeakyReLU(),
           nn.Dropout(dropout),

           # Second block
           nn.Linear(hidden_size, NB_CLASSES), nn.Softmax(1)]
    clf = nn.Sequential(*clf)

    return cnn_part, maxpool_cnn, clf
