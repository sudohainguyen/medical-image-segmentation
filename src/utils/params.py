# Copyright (c) 2020 Hai Nguyen
# 
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT


class Params(object):
    def __init__(self, epochs, lr, seed, **kwargs):
        self.epochs = epochs
        self.lr = lr
        self.seed = seed
        
        for k, v in kwargs:
            setattr(self, k, v)
