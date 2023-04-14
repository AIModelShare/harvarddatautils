
import torch
import torch.nn as nn


# Residual Block
class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=False, pooling=False):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ELU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.maxpool = nn.MaxPool2d(2, stride=2)
        self.downsample = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(out_channels)
        )
        self.downsampleOrNot = downsample
        self.pooling = pooling
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        # out = self.dropout(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsampleOrNot:
            residual = self.downsample(x)
        out += residual
        if self.pooling:
            out = self.maxpool(out)
        out = self.dropout(out)
        return out

def torch_stft(X_train, hop_length_mult=4, eps=1e-8):
    signal = []
    for s in range(X_train.shape[1]):
        spectral = torch.stft(X_train[:, s, :],
                              n_fft=256,
                              hop_length=256 * 1 // hop_length_mult,
                              center=False,
                              onesided=True,
                              return_complex=True)
        spectral = torch.view_as_real(spectral)
        signal.append(spectral)
    signal = torch.stack(signal)
    signal1 = signal[..., 0].swapaxes(0,1)
    signal2 = signal[..., 1].swapaxes(0,1)

    return torch.cat([torch.log(torch.abs(signal1) + eps), torch.log(torch.abs(signal2) + eps)], dim=1)


class CNNEncoder2D_IIIC(nn.Module):
    def __init__(self, n_dim=128, embedding_size=96, nclass=6, num_channels=16, as_encoder=False):
        super(CNNEncoder2D_IIIC, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(2 * num_channels, 3 * num_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(3 * num_channels),
            nn.ELU(inplace=True),
        )
        self.conv2 = ResBlock(3 * num_channels, 4 * num_channels, 2, True, False)
        self.conv3 = ResBlock(4 * num_channels, 8 * num_channels, 2, True, True)
        self.conv4 = ResBlock(8 * num_channels, 16 * num_channels, 2, True, True)
        self.n_dim = n_dim

        self.sup = nn.Sequential(
            nn.Linear(64 * num_channels, embedding_size, bias=True),
            nn.ReLU(),
            nn.Linear(embedding_size, nclass, bias=True),
        )
        self.as_encoder = as_encoder

    def get_readout_layer(self):
        return self.sup[2]

    def delete_readout_layer(self):
        self.sup[2] = torch.nn.Identity()
        return True

    def get_embed_size(self):
        return self.sup[2].in_features

    def forward(self, x, embed_only=False, **kwargs):
        x = x.float()
        x = torch_stft(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = x.reshape(x.shape[0], -1)
        if embed_only or self.as_encoder:
            return self.sup[1](self.sup[0](x))
        return self.sup(x)


