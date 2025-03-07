import torch.nn as nn
import torch

class ConvBlock(nn.Module):
    # Conv -> BN -> leakyReLU
    def __init__(self, in_channels, out_channels, use_activation=True, use_BatchNorm=True, **kwargs):
        super().__init__()
        self.use_activation = use_activation
        self.use_BatchNorm = use_BatchNorm
        self.cnn = nn.Conv2d(in_channels, out_channels, **kwargs)
        if use_BatchNorm:
            self.bn = nn.BatchNorm2d(out_channels)
        self.ac = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        x = self.cnn(x)
        if self.use_BatchNorm:
            x = self.bn(x)
        x1 = self.ac(x)
        return x1 if self.use_activation else x

class UpsampleBlock(nn.Module):
    def __init__(self, in_channels, scale_factor):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, in_channels * (scale_factor**2), kernel_size=2, stride=1, padding=1)
        self.ps = nn.PixelShuffle(scale_factor)
        self.ac = nn.PReLU(num_parameters=in_channels)

    def forward(self, x):
        return self.ac(self.ps(self.conv(x)))

class ResidualBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.b1 = ConvBlock(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
        self.b2 = ConvBlock(in_channels, in_channels, kernel_size=3, stride=1, padding=1, use_activation=False)

    def forward(self, x0):
        x = self.b1(x0)
        x = self.b2(x)
        return x + x0

class Generator(nn.Module):
    def __init__(self, in_channels=3, num_channels=64, num_blocks=8):
        super().__init__()
        self.initial = ConvBlock(in_channels, num_channels, kernel_size=7, stride=1, padding=4, use_BatchNorm=False)
        self.res = nn.Sequential(*[ResidualBlock(num_channels) for i in range(num_blocks)])
        self.conv = ConvBlock(num_channels, num_channels, kernel_size=3, stride=1, padding=1, use_activation=False)
        self.up = nn.Sequential(UpsampleBlock(num_channels, scale_factor=2))
        self.final = nn.Conv2d(num_channels, in_channels, kernel_size=9, stride=1, padding=1)

    def forward(self, x):
        x = self.initial(x)
        c = self.res(x)
        c = self.conv(c) + x
        c = self.up(c)
        c = self.final(c)
        return torch.sigmoid(c)

class Discriminator(nn.Module):
    def __init__(self, in_channels=3, features=[64, 64, 128, 128, 256, 256, 512, 512]):
        super().__init__()
        blocks = []
        for idx, feature in enumerate(features):
            blocks.append(
                ConvBlock(
                    in_channels,
                    feature,
                    kernel_size=3,
                    stride = idx % 2 + 1,
                    padding=1,
                    use_activation=True,
                    use_BatchNorm=idx != 0,
                )
            )
            in_channels = feature

        self.blocks = nn.Sequential(*blocks)

        self.mlp = nn.Sequential(
            nn.AdaptiveAvgPool2d((8, 8)),
            nn.Flatten(),
            nn.Linear(512*8*8, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 1)
        )

    def forward(self, x):
        x = self.blocks(x)
        x = self.mlp(x)
        return x