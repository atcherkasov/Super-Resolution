import torch.nn as nn
import functools


"""
Source: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/networks.py
"""
class NLayerDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator"""

    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d):
        """Construct a PatchGAN discriminator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(NLayerDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = 1
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]  # output 1 channel prediction map
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        """Standard forward."""
        return self.model(input)


class LRDiscriminator(nn.Module):
    def __init__(self, channels_img, features_d):
        super(LRDiscriminator, self).__init__()
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
                features_d, features_d * 2, kernel_size=4, stride=1, padding=1
            ),
            nn.LeakyReLU(0.2),

            # 3
            nn.Conv2d(
                features_d * 2, features_d * 4, kernel_size=4, stride=1, padding=1
            ),
            nn.LeakyReLU(0.2),

            # 4
            nn.Conv2d(
                features_d * 4, features_d * 8, kernel_size=4, stride=1, padding=1
            ),
            nn.LeakyReLU(0.2),

            # 5
            nn.Conv2d(
                features_d * 8, 1, kernel_size=4, stride=1, padding=0
            ),
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


class HRDiscriminator(nn.Module):
    def __init__(self, channels_img, features_d=64):
        super(HRDiscriminator, self).__init__()
        self.disc = nn.Sequential(
            # input: N x channels_img x 32 x 32
            # todo сколько должно быть фильтров на слоях и какого размера?
            # 1
            nn.Conv2d(
                channels_img, features_d, kernel_size=4, stride=2, padding=1
            ),
            nn.LeakyReLU(0.2),

            # 2
            nn.Conv2d(
                features_d, features_d * 2, kernel_size=4, stride=1, padding=1
            ),
            nn.LeakyReLU(0.2),

            # 3
            nn.Conv2d(
                features_d * 2, features_d * 4, kernel_size=4, stride=1, padding=1
            ),
            nn.LeakyReLU(0.2),

            # 4
            nn.Conv2d(
                features_d * 4, features_d * 8, kernel_size=4, stride=1, padding=1
            ),
            nn.LeakyReLU(0.2),

            # 5
            nn.Conv2d(
                features_d * 8, 1, kernel_size=4, stride=1, padding=0
            ),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.disc(x)