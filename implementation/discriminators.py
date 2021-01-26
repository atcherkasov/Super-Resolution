import torch
import torch.nn as nn


class DiscriminatorX(nn.Module):
    def __init__(self, channels_img, features_d):
        super(DiscriminatorX, self).__init__()
        self.disc = nn.Sequential(
            # input: N x channels_img x 32 x 32
            # todo сколько должно быть фильтров на слоях и какого размера?
            # 1
            nn.Conv2d(
                channels_img, features_d, kernel_size=4, stride=1, padding=1
            ),
            nn.LeakyReLU(0.2),

            # 2
            nn.Conv2d(
                channels_img, features_d, kernel_size=4, stride=1, padding=1
            ),
            nn.LeakyReLU(0.2),

            # 3
            nn.Conv2d(
                channels_img, features_d, kernel_size=4, stride=1, padding=1
            ),
            nn.LeakyReLU(0.2),

            # 4
            nn.Conv2d(
                channels_img, features_d, kernel_size=4, stride=1, padding=1
            ),
            nn.LeakyReLU(0.2),

            # 5
            nn.Conv2d(
                channels_img, features_d, kernel_size=4, stride=1, padding=1
            ),
            nn.LeakyReLU(0.2),


            # _block(in_channels, out_channels, kernel_size, stride, padding)
            self._block(features_d, features_d * 2, 4, 2, 1),
            self._block(features_d * 2, features_d * 4, 4, 2, 1),
            # self._block(features_d * 4, features_d * 8, 4, 2, 1),
            # After all _block img output is 4x4 (Conv2d below makes into 1x1)
            nn.Conv2d(features_d * 4, 1, kernel_size=4, stride=2, padding=0),
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
            # nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2),
        )

    def forward(self, x):
        return self.disc(x)