# Copyright (c) 2020 Hai Nguyen
# 
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

import torch.nn as nn
from ...utils import get_activation_layer


class Conv2D(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        activation='relu',
        use_pad=False,
        use_bn=True,
        bn_eps=1e-5
    ):
        super(Conv2D, self).__init__()
        self.use_bn = use_bn
        self.use_pad = use_pad
        self.activation = activation

        if self.use_pad:
            assert padding != 0
            self.pad = nn.ZeroPad2d(padding)
            padding = 0

        self.conv = nn.Conv2d(in_channels, out_channels, 
                              kernel_size=kernel_size, bias=(not use_bn),
                              groups=groups, padding=padding)

        if self.use_bn:
            self.bn = nn.BatchNorm2d(out_channels, eps=bn_eps)

        if self.activation:
            self.act = get_activation_layer(self.activation)

    def forward(self, x):
        if self.use_pad:
            x = self.pad(x)
        x = self.conv(x)
        if self.use_bn:
            x = self.bn(x)
        if self.activation:
            x = self.act(x)
        return x


def conv2d_1x1_relu(in_channels, out_channels, use_bn=True):
    return Conv2D(in_channels, out_channels, kernel_size=1, use_bn=use_bn, 
                  stride=1, activation='relu')


def conv2d_3x3_relu(in_channels, out_channels, use_bn=True):
    return Conv2D(in_channels, out_channels, kernel_size=3, use_bn=use_bn,
                  padding=1, stride=1, activation='relu')


def conv2d_5x5_relu(in_channels, out_channels):
    return Conv2D(in_channels, out_channels, kernel_size=5,
                  padding=2, stride=1, activation='relu')
    

def conv2d_1x1(in_channels, out_channels, stride=1, use_bn=True):
    return Conv2D(in_channels, out_channels, kernel_size=1, use_bn=use_bn,
                  stride=stride, activation=None)

    
def conv2d_3x3(in_channels, out_channels, stride=1, use_bn=True):
    return Conv2D(in_channels, out_channels, kernel_size=3, use_bn=use_bn,
                  padding=1, stride=stride, activation=None)


class GlobalAveragePooling(nn.Module):
    def __init__(self):
        super(GlobalAveragePooling, self).__init__()
    
    def forward(self, x):
        return nn.functional.adaptive_avg_pool2d(x, 1).view(x.size(0), -1)
