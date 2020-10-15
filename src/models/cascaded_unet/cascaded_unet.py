# Copyright (c) 2020 Hai Nguyen
# 
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

import torch
import torch.nn as nn

from .core import EncoderBlock, DecoderBlock
from ..blocks.common import conv2d_1x1, conv2d_3x3_relu


class UNetBlock(nn.Module):
    def __init__(
        self,
        mid_channels,
        in_channels=None,
        depth=2,
    ):
        super(UNetBlock, self).__init__()
        channels = [mid_channels // (2 ** i) for i in range(depth, -1, -1)]
        self.d = depth
        
        if not in_channels:
            in_channels = channels[0]
        
        self.mid = nn.Sequential(
            conv2d_3x3_relu(mid_channels // 2, mid_channels),
            conv2d_3x3_relu(mid_channels, mid_channels)
        )
        self.enc0 = EncoderBlock(in_channels, channels[0])
        self.dec0 = DecoderBlock(channels[1], channels[0], channels[1])
        self.pool0 = nn.MaxPool2d(kernel_size=2)
        for d in range(1, self.d):
            setattr(self, f'enc{d}',
                    EncoderBlock(channels[d - 1], channels[d]))
            setattr(self, f'dec{d}', 
                    DecoderBlock(channels[d + 1], channels[d], channels[d + 1]))
            setattr(self, f'pool{d}', nn.MaxPool2d(kernel_size=2))

    def forward(self, x):
        skips = []
        for d in range(0, self.d):
            x = getattr(self, f'enc{d}')(x)
            skips.append(x)
            x = getattr(self, f'pool{d}')(x)
        x = self.mid(x)
        for d in range(self.d - 1, -1, -1):
            x = getattr(self, f'dec{d}')(x, skips.pop())
        return x


class CascadedUNet(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        block_depth: int = 2,
        block_mid_channels: int = 128,
        net_len: int = 4
    ):
        super(CascadedUNet, self).__init__()
        self.net_len = net_len
        unet_out_channels = block_mid_channels // (2 ** block_depth)
        
        self.unet_0 = UNetBlock(in_channels, unet_out_channels)

        for i in range(1, self.net_len):
            setattr(self, f'unet_{i}', UNetBlock(
                in_channels=unet_out_channels,
                mid_channels=block_mid_channels,
                depth=block_depth)
            )
        
        self.last_conv = conv2d_1x1(
            in_channels=unet_out_channels,
            out_channels=out_channels
        )

    def forward(self, x):
        x = self.unet_0(x)
        for i in range(1, self.net_len):
            x = getattr(self, f'unet_{i}')(x)
        x = self.last_conv(x)
        x = torch.sigmoid(x)
        return x
