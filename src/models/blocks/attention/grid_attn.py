# Copyright (c) 2020 Hai Nguyen
# 
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

import torch.nn as nn


class GridAttention(nn.Module):
    def __init__(self):
        super(GridAttention, self).__init__()
