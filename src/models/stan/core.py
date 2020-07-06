# Copyright (c) 2020 Hai Nguyen
# 
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..blocks.common import conv2d_5x5_relu, conv2d_1x1_relu, conv2d_3x3_relu


class EncoderBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels
    ):
        super(EncoderBlock, self).__init__()
        
        self.conv3_1 = conv2d_3x3_relu(in_channels, out_channels)
        self.conv3_2 = conv2d_3x3_relu(out_channels, out_channels)
        self.pool3 = nn.MaxPool2d(kernel_size=2)
        
        out_channels = out_channels // 2
        # if in_channels != 3:
        #     in_channels *= 2
        self.conv1_1 = conv2d_1x1_relu(in_channels, out_channels)
        self.conv1_2 = conv2d_1x1_relu(out_channels, out_channels)

        self.conv5_1 = conv2d_5x5_relu(in_channels, out_channels)
        self.conv5_2 = conv2d_5x5_relu(out_channels, out_channels)

        self.pool_concat = nn.MaxPool2d(kernel_size=2)

    def forward(self, kernel3_inp, kernelconcat_inp):
        x1 = self.conv1_1(kernelconcat_inp)
        x1 = self.conv1_2(x1)

        x5 = self.conv5_1(kernelconcat_inp)
        x5 = self.conv5_2(x5)
        concat = torch.cat([x1, x5], dim=1)
        concat_pool = self.pool_concat(concat)

        x3_1 = self.conv3_1(kernel3_inp)
        skip1 = self.conv3_2(x3_1)
        x3_pool = self.pool3(skip1)

        skip2 = torch.add(x3_1, concat)
        # skip2 = torch.cat([x3_1, concat], dim=1)
        return x3_pool, concat_pool, skip1, skip2


class DecoderBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        skip_channels,
        # skip1_channels,
        # skip2_channels,
        output_size=None,
        use_bn=False,
        mode='transpose'
    ):
        super(DecoderBlock, self).__init__()
        assert mode in ['transpose', 'upsampling']
        
        self.use_bn = use_bn
        if mode == 'transpose':
            self.up = nn.ConvTranspose2d(in_channels, out_channels,
                                         padding=1, kernel_size=3, stride=2)
            self.output_size = output_size
        else:
            self.up = nn.Upsample(size=2)

        if self.use_bn:
            self.bn = nn.BatchNorm2d(out_channels)
        
        self.conv_1 = conv2d_3x3_relu(in_channels=out_channels + skip_channels,
                                      out_channels=out_channels)
        self.conv_2 = conv2d_3x3_relu(in_channels=out_channels + skip_channels,
                                      out_channels=out_channels))
        # self.conv_1 = conv2d_3x3_relu(out_channels + skip1_channels,
        #                               out_channels)
        # self.conv_2 = conv2d_3x3_relu(out_channels + skip2_channels,
        #                               out_channels)

    def forward(self, inp, skip1, skip2):
        x = self.up(inp, output_size=self.output_size)
        if self.use_bn:
            x = self.bn(x)
        x = F.relu(x)
        concat = torch.cat([x, skip1], dim=1)
        x = self.conv_1(concat)
        concat = torch.cat([x, skip2], dim=1)
        x = self.conv_2(concat)
        return x


class MidBlock(nn.Module):
    def __init__(self, num_channels):
        super(MidBlock, self).__init__()

        self.conv3_1 = conv2d_3x3_relu(num_channels // 2, num_channels)
        self.conv3_2 = conv2d_3x3_relu(num_channels, num_channels)
        
        self.conv1_1 = conv2d_1x1_relu(num_channels // 2, num_channels)
        # self.conv1_1 = conv2d_1x1_relu(num_channels, num_channels)
        self.conv1_2 = conv2d_3x3_relu(num_channels, num_channels)
        
        self.conv5_1 = conv2d_5x5_relu(num_channels // 2, num_channels)
        # self.conv5_1 = conv2d_5x5_relu(num_channels, num_channels)
        self.conv5_2 = conv2d_3x3_relu(num_channels, num_channels)
    
    def forward(self, x3, concat):
        x3 = self.conv3_1(x3)
        x3 = self.conv3_2(x3)
        
        x1 = self.conv1_1(concat)
        x1 = self.conv1_2(x1)
        
        x5 = self.conv5_1(concat)
        x5 = self.conv5_2(x5)
        return torch.cat([x3, x1, x5], dim=1)
