# Copyright (c) 2020 Hai Nguyen
# 
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

import torch.nn as nn

from .core import EncoderBlock, DecoderBlock, MidBlock
from ..blocks.common import Conv2D


class STAN(nn.Module):
    """ Small tumor-aware network (STAN) implementation
        Ref: https://arxiv.org/pdf/2002.01034.pdf
    """    
    def __init__(
        self,
        in_channels=3,
        out_channels=1,
        input_size=256,
        mid_channels=512,
        decode_mode='transpose',
        activation_out='sigmoid'
    ):
        """STAN initializer

        Parameters
        ----------
        in_channels : int, optional
            Input channels, by default 3
        out_channels : int, optional
            Output channels, by default 1
        input_size : int, optional
            Input size of image - (input_size, input_size), by default 256
        mid_channels : int, optional
            Number filters in conv layers at mid block, by default 512
        decode_mode : str, optional
            Could be transpose or upsampling, by default 'transpose'
        activation_out : str, optional
            Activation function name for output prediction,
            by default 'sigmoid'
        """    
        super(STAN, self).__init__()
        self.channels = [mid_channels // (2 ** i) 
                         for i in range(4, 0, -1)]
        self.enc_1 = EncoderBlock(in_channels, self.channels[0])
        self.enc_2 = EncoderBlock(self.channels[0], self.channels[1])
        self.enc_3 = EncoderBlock(self.channels[1], self.channels[2])
        self.enc_4 = EncoderBlock(self.channels[2], self.channels[3])

        self.mid = MidBlock(mid_channels)

        self.dec_4 = DecoderBlock(
            mid_channels * 3, self.channels[3], 
            self.channels[3], mode=decode_mode, 
            output_size=(input_size // (2 ** 3), input_size // (2 ** 3))
        )
        self.dec_3 = DecoderBlock(
            self.channels[3], self.channels[2],
            self.channels[2], mode=decode_mode,
            output_size=(input_size // (2 ** 2), input_size // (2 ** 2))
        )
        self.dec_2 = DecoderBlock(
            self.channels[2], self.channels[1],
            self.channels[1], mode=decode_mode,
            output_size=(input_size // (2 ** 1), input_size // (2 ** 1))
        )
        self.dec_1 = DecoderBlock(
            self.channels[1], self.channels[0],
            self.channels[0], mode=decode_mode,
            output_size=(input_size, input_size)
        )
        self.conv_out = Conv2D(self.channels[0], out_channels,
                               kernel_size=1, activation=activation_out)

    def forward(self, x):
        x3_pool, concat_pool, skip1_b1, skip2_b1 = self.enc_1(x, x)
        x3_pool, concat_pool, skip1_b2, skip2_b2 = self.enc_2(x3_pool, concat_pool)  # noqa: E501
        x3_pool, concat_pool, skip1_b3, skip2_b3 = self.enc_3(x3_pool, concat_pool)  # noqa: E501
        x3_pool, concat_pool, skip1_b4, skip2_b4 = self.enc_4(x3_pool, concat_pool)  # noqa: E501

        mid = self.mid(x3_pool, concat_pool)

        up = self.dec_4(mid, skip1_b4, skip2_b4)
        up = self.dec_3(up, skip1_b3, skip2_b3)
        up = self.dec_2(up, skip1_b2, skip2_b2)
        up = self.dec_1(up, skip1_b1, skip2_b1)

        out = self.conv_out(up)
        return out
