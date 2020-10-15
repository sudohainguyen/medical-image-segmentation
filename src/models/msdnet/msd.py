# Copyright (c) 2020 Hai Nguyen
# 
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT


import torch
import torch.nn as nn


class MSDLayer(nn.Module):
    def __init__(self, in_channels, dilations, kernel_size=3):
        super(MSDLayer, self).__init__()
        # dilations could be int or list
        if type(dilations) == int:
            dilations = [j % 10 + 1 for j in range(dilations)]

        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.out_channels = in_channels + len(dilations)

        for j, dilation in enumerate(dilations):
            # Equal to: kernel_size + (kernel_size - 1) * (dilation - 1)
            dilated_kernel_size = (kernel_size - 1) * dilation + 1
            padding = dilated_kernel_size // 2
            self.add_module(f'conv_{j}', nn.Sequential(
                nn.Conv2d(in_channels, 1, kernel_size=kernel_size,
                          dilation=dilation, padding=padding),
                nn.ReLU(inplace=True)
            )

    def forward(self, x):
        return torch.cat((x,) + tuple(c(x) for c in self.children()), dim=1)


class MSDNet(nn.Sequential):
    def __init__(self, in_channels, out_channels, depth, width, kernel_size=3):
        super(MSDNet, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        current_channels = in_channels
        for i in range(depth):
            dilations = [(i * width + j) % 10 + 1 for j in range(width)]
            layer = MSDLayer(current_channels, dilations, kernel_size)
            current_channels = layer.out_channels
            self.add_module(f'layer_{i}', layer)

        self.add_module('last', nn.Sequential(
            nn.Conv2d(current_channels, out_channels, kernel_size=1),
            nn.Sigmoid()
        )
