# Copyright (c) 2020 Hai Nguyen
# 
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

import math


class LR_Scheduler(object):
    def __init__(
        self, mode, base_lr, num_epochs,
        iters_per_epoch=0, lr_step=0, warmup_epochs=0
    ):
        assert mode in ['step', 'cos', 'poly'], \
            f'The given mode: `{mode}` is not valid'
        
        if mode == 'step':
            assert lr_step
        
        self.mode = mode
        print(f'Using {self.mode} LR scheduler!')
        self.lr = base_lr
        self.lr_step = lr_step
        self.iters_per_epoch = iters_per_epoch
        self.N = num_epochs * iters_per_epoch
        self.epoch = -1
        self.warmup_iters = warmup_epochs * iters_per_epoch

    def __call__(self, optimizer, i, epoch, best_pred):
        T = epoch * self.iters_per_epoch + i
        if self.mode == 'cos':
            lr = 0.5 * self.lr * (1 + math.cos(T / self.N * math.pi))
        elif self.mode == 'poly':
            lr = self.lr * pow(1 - T / self.N, 0.9)
        else:
            lr = self.lr * (0.1 ** (epoch // self.lr_step))
        
        if self.warmup_iters > 0 and T < self.warmup_iters:
            lr = lr * T / self.warmup_iters
        if epoch > self.epoch:
            self.epoch = epoch
        assert lr >= 0
        self._adjust_lr(optimizer, lr)
        
    def _adjust_lr(self, optimizer, lr):
        if len(optimizer.param_groups) == 1:
            optimizer.param_groups[0]['lr'] = lr
            return
        for i in range(len(optimizer.param_groups)):
            if optimizer.param_groups[i]['lr'] > 0:
                optimizer.param_groups[i]['lr'] = lr