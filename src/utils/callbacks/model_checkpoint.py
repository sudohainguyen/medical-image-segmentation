# Copyright (c) 2020 Hai Nguyen
# 
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

import torch


class ModelCheckpoint:
    def __init__(self, verbose=True, path='checkpoint.pth'):
        self.verbose = verbose
        self.path = path
        self.val_loss_min = None

    def step(self, val_loss, model):
        if not self.val_loss_min:
            self.val_loss_min = val_loss
        elif self.val_loss_min > val_loss:
            self.val_loss_min = val_loss
            if self.verbose:
                print(f'Saving best model... with {self.val_loss_min:3f}')
            torch.save(model.state_dict(), self.path)
