# Copyright (c) 2020 Hai Nguyen
# 
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

import torch
import torch.nn as nn
from ..common import conv2d_1x1


class AttentionGate(torch.nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(AttentionGate, self).__init__()
        
        self.W_g = conv2d_1x1(in_channels=F_g, out_channels=F_int)
        self.W_x = conv2d_1x1(in_channels=F_l, out_channels=F_int)
        self.bn = nn.BatchNorm2d(F_int)
        self.psi = conv2d_1x1(in_channels=F_int, out_channels=1)
        
    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = torch.relu(torch.add(g1, x1))
        psi = self.bn(psi)
        psi = self.psi(psi)
        psi = torch.sigmoid(psi)
        return x * psi
