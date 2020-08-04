# Copyright (c) 2020 Hai Nguyen
# 
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

import torch
import torch.nn as nn

from .core import ConvBlock, UpConv
from ..blocks.common import conv2d_1x1


class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, 
                 mid_channels=1024, depth=4):
        super(UNet, self).__init__()
        self.depth = depth
        channels = [mid_channels // (2 ** i) for i in range(depth, -1, -1)]
        # 64 128 256 512 1024

        self.enc0 = ConvBlock(in_channels, channels[0])
        self.pool0 = nn.MaxPool2d(kernel_size=2)
        self.up0 = UpConv(channels[1], channels[0])
        self.dec0 = ConvBlock(channels[1], channels[0])
        self.mid = ConvBlock(mid_channels // 2, mid_channels)
        for i in range(1, self.depth):
            setattr(self, f'enc{i}', ConvBlock(channels[i - 1], channels[i]))
            setattr(self, f'pool{i}', nn.MaxPool2d(kernel_size=2))
            setattr(self, f'up{i}', UpConv(channels[i + 1], channels[i]))
            setattr(self, f'dec{i}', ConvBlock(channels[i + 1], channels[i]))
        
        self.conv_out = conv2d_1x1(channels[0], out_channels)

    def forward(self, x):
        # encoding path
        skips = []
        for i in range(self.depth):
            x = getattr(self, f'enc{i}')(x)
            skips.append(x)
            x = getattr(self, f'pool{i}')(x)
        
        x = self.mid(x)

        for i in range(self.depth - 1, -1, -1):
            x = getattr(self, f'up{i}')(x)
            x = torch.cat((x, skips.pop()), dim=1)
            x = getattr(self, f'dec{i}')(x)
        
        x = self.conv_out(x)
        out = torch.sigmoid(x)
        return out
