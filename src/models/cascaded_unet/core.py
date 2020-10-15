# Copyright (c) 2020 Hai Nguyen
# 
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

import torch
import torch.nn as nn

from ..blocks.common import conv2d_3x3_relu
from ..blocks.resnext import ResNeXtUnit


class EncoderBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        use_bn=True
    ):
        super(EncoderBlock, self).__init__()
        # self.core = nn.Sequential(
        #     conv2d_3x3_relu(in_channels, out_channels, use_bn=use_bn),
        #     conv2d_3x3_relu(out_channels, out_channels, use_bn=use_bn),
        # )
        self.core = ResNeXtUnit(in_channels, out_channels, cardinality=4)
    
    def forward(self, x):
        x = self.core(x)
        return x


class DecoderBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        skip_channels,
        use_bn=True
    ):
        super(DecoderBlock, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels,
                                     kernel_size=3, padding=1, stride=2)
        afterup_modules = []
        if use_bn:
            afterup_modules.append(nn.BatchNorm2d(in_channels))
        # afterup_modules.append(nn.ReLU(inplace=True))
        self.afterup = nn.Sequential(*afterup_modules)
        self.core = nn.Sequential(
            conv2d_3x3_relu(in_channels + skip_channels,
                            out_channels, use_bn=use_bn),
            conv2d_3x3_relu(out_channels, out_channels, use_bn=use_bn)
        )

    def forward(self, x, skip):
        x = self.up(x, output_size=skip.shape[2:])
        x = self.afterup(x)
        x = torch.cat([x, skip], dim=1)
        x = self.core(x)
        return x


class PixelShuffle_ICNR(nn.Module):
    """
    Upsample by `scale` from `ni` filters to `nf` (default `ni`), 
    using `nn.PixelShuffle`, `icnr` init, and `weight_norm`.
    """
    def __init__(
        self,
        ni: int,
        nf: int = None,
        scale: int = 2,
        blur: bool = False,
        leaky: float = None
    ):
        super(PixelShuffle_ICNR, self).__init__()
        # nf = ifnone(nf, ni)
        nf = ni if nf is None else ni
        # self.conv = conv2d_1x1(ni, nf * (scale ** 2))
        self.conv = nn.Conv2d(ni, nf * (scale ** 2), kernel_size=1)
        # icnr(self.conv[0].weight)
        self.shuf = nn.PixelShuffle(scale)
        # Blurring over (h*w) kernel
        # "Super-Resolution using Convolutional Neural Networks 
        #  without AnyCheckerboard Artifacts"
        # - https://arxiv.org/abs/1806.02658
        self.pad = nn.ReplicationPad2d((1, 0, 1, 0))
        self.blur = nn.AvgPool2d(2, stride=1)
        self.do_blur = blur
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.shuf(self.relu(self.conv(x)))
        return self.blur(self.pad(x)) if self.do_blur else x


def icnr(x, scale=2, init=nn.init.kaiming_normal_):
    "ICNR init of `x`, with `scale` and `init` function."
    ni, nf, h, w = x.shape
    ni2 = int(ni / (scale ** 2))
    k = init(torch.zeros([ni2, nf, h, w])).transpose(0, 1)
    k = k.contiguous().view(ni2, nf, -1)
    k = k.repeat(1, 1, scale ** 2)
    k = k.contiguous().view([nf, ni, h, w]).transpose(0, 1)
    x.data.copy_(k)
