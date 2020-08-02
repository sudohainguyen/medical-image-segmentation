# Copyright (c) 2020 Hai Nguyen
# 
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

import torch
import torch.nn as nn
from .common import conv2d_3x3_relu


class SEBlock(nn.Module):
    def __init__(self, n_channels, reduction=16):
        super(SEBlock, self).__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.se = nn.Sequential(
            nn.Linear(n_channels, n_channels // reduction, bias=False),
            nn.BatchNorm1d(n_channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(n_channels // reduction, n_channels, bias=False),
            nn.BatchNorm1d(n_channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.gap(x).view(b, c)
        y = self.se(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class SEResBlock(nn.Module):
    def __init__(self, n_channels, reduction=16):
        super(SEResBlock, self).__init__()
        self.residual = nn.Sequential(
            conv2d_3x3_relu(n_channels, n_channels, use_bn=True),
            conv2d_3x3_relu(n_channels, n_channels, use_bn=True)
        )
        self.sebranch = SEBlock(n_channels, reduction)
    
    def forward(self, x):
        x = self.residual(x)
        identity = x
        x = self.sebranch(x)
        x = torch.add(x, identity)
        x = torch.relu(x)
        return x
