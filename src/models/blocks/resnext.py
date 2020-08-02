# Copyright (c) 2020 Hai Nguyen
# 
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

import math
import torch
import torch.nn as nn
from .common import (
    conv2d_1x1_relu, conv2d_1x1, Conv2D
)


class ResNeXtBottleneck(nn.Module):
    """
    ResNeXt bottleneck block for residual path in ResNeXt unit.
    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    stride : int or tuple/list of 2 int
        Strides of the convolution.
    cardinality: int
        Number of groups.
    bottleneck_width: int
        Width of bottleneck block.
    bottleneck_factor : int, default 4
        Bottleneck factor.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 stride=1,
                 cardinality=32,
                 bottleneck_width=4,
                 bottleneck_factor=4):
        super(ResNeXtBottleneck, self).__init__()

        mid_channels = out_channels // bottleneck_factor
        D = int(math.floor(mid_channels * (bottleneck_width / 64.0)))
        group_width = cardinality * D
        self.core = nn.Sequential(
            conv2d_1x1_relu(in_channels, group_width),
            Conv2D(group_width, group_width, kernel_size=3, padding=1,
                   stride=stride, groups=cardinality, activation='relu'),
            conv2d_1x1(group_width, out_channels),
        )

    def forward(self, x):
        x = self.core(x)
        return x


class ResNeXtUnit(nn.Module):
    """
    ResNeXt unit with residual connection.
    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    stride : int or tuple/list of 2 int
        Strides of the convolution.
    cardinality: int
        Number of groups.
    bottleneck_width: int
        Width of bottleneck block.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 stride=1,
                 cardinality=32,
                 bottleneck_width=4):
        super(ResNeXtUnit, self).__init__()
        self.resize_identity = (in_channels != out_channels) or (stride != 1)

        self.body = ResNeXtBottleneck(
            in_channels=in_channels,
            out_channels=out_channels,
            stride=stride,
            cardinality=cardinality,
            bottleneck_width=bottleneck_width)
        
        if self.resize_identity:
            self.identity_conv = conv2d_1x1(
                in_channels=in_channels,
                out_channels=out_channels,
                stride=stride)

    def forward(self, x):
        if self.resize_identity:
            identity = self.identity_conv(x)
        else:
            identity = x
        x = self.body(x)
        x = torch.add(x, identity)
        x = torch.relu(x)
        return x
