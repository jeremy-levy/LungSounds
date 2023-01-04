import torch
import torch.nn as nn
import torch.nn.functional as F
from torchaudio import transforms as T


def init_layer(layer):
    """Initialize a Linear or Convolutional layer. """
    nn.init.xavier_uniform_(layer.weight)
    if hasattr(layer, 'bias'):
        if layer.bias is not None:
            layer.bias.data.fill_(0.)


def init_bn(bn):
    """Initialize a Batchnorm layer. """
    bn.bias.data.fill_(0.)
    bn.weight.data.fill_(1.)


class ConvBlock(nn.Module):  # for CNN10 & CNN14
    def __init__(self, in_channels, out_channels, kernel_size, do_dropout):

        super(ConvBlock, self).__init__()

        self.dropout = nn.Dropout(0.2)
        self.do_dropout = do_dropout

        self.conv1 = nn.Conv2d(in_channels=in_channels,
                               out_channels=out_channels,
                               kernel_size=(3, kernel_size), stride=(1, 1),
                               padding=(1, 1), bias=False)

        self.conv2 = nn.Conv2d(in_channels=out_channels,
                               out_channels=out_channels,
                               kernel_size=(3, kernel_size), stride=(1, 1),
                               padding=(1, 1), bias=False)

        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.init_weight()

    def init_weight(self):
        init_layer(self.conv1)
        init_layer(self.conv2)
        init_bn(self.bn1)
        init_bn(self.bn2)

    def forward(self, input, pool_size=(2, 2), pool_type='avg'):
        x = input
        x = F.relu_(self.bn1(self.conv1(x)))
        x = F.relu_(self.bn2(self.conv2(x)))
        if pool_type == 'max':
            x = F.max_pool2d(x, kernel_size=pool_size)
        elif pool_type == 'avg':
            x = F.avg_pool2d(x, kernel_size=pool_size)
        elif pool_type == 'avg+max':
            x1 = F.avg_pool2d(x, kernel_size=pool_size)
            x2 = F.max_pool2d(x, kernel_size=pool_size)
            x = x1 + x2
        else:
            raise Exception('Incorrect argument!')

        if self.do_dropout:
            x = self.dropout(x)
        return x


class CNN14(nn.Module):
    def __init__(self, kernel_size, pool_type, num_classes=4, do_dropout=False, embed_only=False):
        super(CNN14, self).__init__()

        self.embed_only = embed_only
        self.pool_type = pool_type
        self.conv_block1 = ConvBlock(in_channels=1, out_channels=64, kernel_size=kernel_size, do_dropout=do_dropout)
        self.conv_block2 = ConvBlock(in_channels=64, out_channels=128, kernel_size=kernel_size, do_dropout=do_dropout)
        self.conv_block3 = ConvBlock(in_channels=128, out_channels=256, kernel_size=kernel_size, do_dropout=do_dropout)
        self.conv_block4 = ConvBlock(in_channels=256, out_channels=512, kernel_size=kernel_size, do_dropout=do_dropout)
        self.conv_block5 = ConvBlock(in_channels=512, out_channels=1024, kernel_size=kernel_size, do_dropout=do_dropout)
        self.conv_block6 = ConvBlock(in_channels=1024, out_channels=2048, kernel_size=kernel_size,
                                     do_dropout=do_dropout)

        self.linear = nn.Linear(2048, num_classes, bias=True)

    def forward(self, x):
        x = self.conv_block1(x, pool_size=(2, 2), pool_type=self.pool_type)

        x = self.conv_block2(x, pool_size=(2, 2), pool_type=self.pool_type)
        x = self.conv_block3(x, pool_size=(2, 2), pool_type=self.pool_type)
        x = self.conv_block4(x, pool_size=(2, 2), pool_type=self.pool_type)
        x = self.conv_block5(x, pool_size=(2, 2), pool_type=self.pool_type)
        x = self.conv_block6(x, pool_size=(1, 1), pool_type=self.pool_type)

        x = torch.mean(x, dim=3)  # mean over time dim
        (x1, _) = torch.max(x, dim=2)  # max over freq dim
        x2 = torch.mean(x, dim=2)  # mean over freq dim (after mean over time)
        x = x1 + x2

        if self.embed_only:
            return x
        return self.linear(x)


class Normalize(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.power_to_db = T.AmplitudeToDB()

    def forward(self, x):
        x = self.power_to_db(x)
        return (x - x.min()) / (x.max() - x.min())


class Standardize(torch.nn.Module):
    def __init__(self, mean=0.3690, std=0.0255, device='cpu'):  # official split mean & std
        super().__init__()
        self.mean = mean
        self.std = std
        self.device = device

    def forward(self, x):
        return (x - self.mean) / self.std


class SpecAugment(torch.nn.Module):
    def __init__(self, freq_mask=20, time_mask=50, freq_stripes=2, time_stripes=2, p=1.0):
        super().__init__()
        self.p = p
        self.freq_mask = freq_mask
        self.time_mask = time_mask
        self.freq_stripes = freq_stripes
        self.time_stripes = time_stripes
        self.specaugment = nn.Sequential(
            *[T.FrequencyMasking(freq_mask_param=self.freq_mask, iid_masks=True) for _ in range(self.freq_stripes)],
            *[T.TimeMasking(time_mask_param=self.time_mask, iid_masks=True) for _ in range(self.time_stripes)],
        )

    def forward(self, audio):
        if self.p > torch.randn(1):
            return self.specaugment(audio)
        else:
            return audio


def get_mel_transform(add_augmentation):
    melspec = T.MelSpectrogram(n_fft=1024, n_mels=64, win_length=1024, hop_length=512, f_min=50, f_max=2000)
    normalize = Normalize()
    melspec = torch.nn.Sequential(melspec, normalize)
    # standardize = Standardize()

    # Data transformations
    specaug = SpecAugment(freq_mask=20, time_mask=40, freq_stripes=2, time_stripes=2)

    val_transform = nn.Sequential(melspec)
    if add_augmentation is True:
        train_transform = nn.Sequential(melspec, specaug)
    else:
        train_transform = nn.Sequential(melspec)

    return train_transform, val_transform
