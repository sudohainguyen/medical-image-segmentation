# Copyright (c) 2020 Hai Nguyen
# 
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

import torch
import torch.nn as nn
from ..blocks.common import Conv2D


class MixedScaleDenseLayer(nn.Module):
    def __init__(self, in_channels, dilations, kernel_size=3):
        super(MixedScaleDenseLayer, self).__init__()

        if type(dilations) == int:
            dilations = [j % 10 + 1 for j in range(dilations)]

        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.out_channels = in_channels + len(dilations)

        for j, dilation in enumerate(dilations):
            # Equal to: kernel_size + (kernel_size - 1) * (dilation - 1)
            dilated_kernel_size = (kernel_size - 1) * dilation + 1
            padding = dilated_kernel_size // 2
            self.add_module(f'conv_{j}', Conv2D(
                in_channels, 1,
                kernel_size=kernel_size, dilation=dilation, padding=padding
            ))

    def forward(self, x):
        return torch.cat((x,) + tuple(c(x) for c in self.children()), dim=1)
