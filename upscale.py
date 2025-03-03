import numpy as np
import cv2
import torch.nn as nn
import torch

filename = 'im2.jpg'
im = cv2.imread(filename)
im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
im = im[170:830, 350:1200, :]
print(im.shape)

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


gen = Generator()
gen_state_dict = torch.load('gen.pth', weights_only=True, map_location=torch.device('cpu'))
gen.load_state_dict(gen_state_dict)

x = torch.tensor(im).permute(2, 0, 1) / 255
gen.eval()
up = gen(x.unsqueeze(0))
im_up = up.squeeze().permute(1, 2, 0)
im_up = np.array(im_up.detach())
im_up = np.uint8(im_up*255)
print('Writing...')
cv2.imwrite(f'{filename.split(".")[0]}_upscaled.{filename.split(".")[1]}', cv2.cvtColor(im_up, cv2.COLOR_RGB2BGR))
