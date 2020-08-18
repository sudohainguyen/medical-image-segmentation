# Copyright (c) 2020 Hai Nguyen
# 
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

"""
UNet and its variants implementation
"""

import torch.nn as nn

from ..blocks.common import conv2d_3x3_relu


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, use_bn=True):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            conv2d_3x3_relu(in_channels, out_channels, use_bn=use_bn),
            conv2d_3x3_relu(out_channels, out_channels, use_bn=use_bn)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class UpConv(nn.Module):
    def __init__(self, in_channels, out_channels, use_bn=True):
        super(UpConv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            conv2d_3x3_relu(in_channels, out_channels, use_bn=use_bn)
        )

    def forward(self, x):
        x = self.up(x)
        return x  


class UpTranspose(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpTranspose, self).__init__()
        self.up = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, 
                               kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.up(x)
