"""
All models
"""

import cv2
import torch
import torch.nn as nn


def Downscale(img, dim, sigma):
    """
    Downscaling source/target y ∈ Y domain -->
    :param img:
    :param target_size:
    :return:
    """
    # todo set correct sigma and kernel size
    blured = cv2.GaussianBlur(img, (19, 19), 0)
    resized = cv2.resize(blured, dim, interpolation=cv2.INTER_AREA)

    return resized


class DiscriminatorX(nn.Module):
    def __init__(self, channels_img, features_d):
        super(DiscriminatorX, self).__init__()
        self.disc = nn.Sequential(
            # input: N x channels_img x 64 x 64
            nn.Conv2d(
                channels_img, features_d, kernel_size=4, stride=2, padding=1
            ),
            nn.LeakyReLU(0.2),
            # _block(in_channels, out_channels, kernel_size, stride, padding)
            self._block(features_d, features_d * 2, 4, 2, 1),
            self._block(features_d * 2, features_d * 4, 4, 2, 1),
            self._block(features_d * 4, features_d * 8, 4, 2, 1),
            # After all _block img output is 4x4 (Conv2d below makes into 1x1)
            nn.Conv2d(features_d * 8, 1, kernel_size=4, stride=2, padding=0),
            nn.Sigmoid(),
        )

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                bias=False,
            ),
            #nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2),
        )

    def forward(self, x):
        return self.disc(x)


class GeneratorYX(nn.Module):
    def __init__(self, channels_noise, channels_img, features_g):
        super(GeneratorYX, self).__init__()
        self.net = nn.Sequential(
            # Input: N x channels_noise x 1 x 1
            self._block(channels_noise, features_g * 16, 4, 1, 0),  # img: 4x4
            self._block(features_g * 16, features_g * 8, 4, 2, 1),  # img: 8x8
            self._block(features_g * 8, features_g * 4, 4, 2, 1),  # img: 16x16
            self._block(features_g * 4, features_g * 2, 4, 2, 1),  # img: 32x32
            nn.ConvTranspose2d(
                features_g * 2, channels_img, kernel_size=4, stride=2, padding=1
            ),
            # Output: N x channels_img x 64 x 64
            nn.Tanh(),
        )

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.ConvTranspose2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                bias=False,
            ),
            #nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x)


def initialize_weights(model):
    # Initializes weights according to the DCGAN paper
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)

def test():
    N, in_channels, H, W = 8, 3, 64, 64
    noise_dim = 100
    x = torch.randn((N, in_channels, H, W))
    disc = DiscriminatorX(in_channels, 8)
    assert disc(x).shape == (N, 1, 1, 1), "Discriminator test failed"
    gen = GeneratorYX(noise_dim, in_channels, 8)
    z = torch.randn((N, noise_dim, 1, 1))
    assert gen(z).shape == (N, in_channels, H, W), "Generator test failed"


# test()


class GeneratorXY(nn.Module):
    def __init__(self, channels_noise, channels_img, features_g):
        super(GeneratorXY, self).__init__()

    def forward(self, x):
        return self.net(x)


class DiscriminatorY(nn.Module):
    def __init__(self, channels_img, features_d):
        super(DiscriminatorY, self).__init__()

    def forward(self, x):
        return self.disc(x)


class DiscriminatorBigX(nn.Module):
    def __init__(self, channels_img, features_d):
        super(DiscriminatorBigX, self).__init__()

    def forward(self, x):
        return self.disc(x)