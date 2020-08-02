# Copyright (c) 2020 Hai Nguyen
# 
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...utils import to_one_hot


class FocalLoss(nn.Module):
    def __init__(self, gamma=0, eps=1e-7, onehot=True, 
                 ignore=None, size_avg=True):
        """Focal Loss function introduced in
            https://arxiv.org/abs/1708.02002

        Parameters
        ----------
        gamma : int, optional
            Gamma parameter, by default 0
        eps : [type], optional
            Epsilon, by default 1e-7
        onehot : bool, optional
            To convert label to onehot or not, by default True
        ignore : [type], optional
            Ignore background class, usually 0, by default None
        size_avg : bool, optional
            Average the loss by size, by default True
        """
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.eps = eps
        self.onehot = onehot
        self.ignore = ignore
        self.size_avg = size_avg

    def forward(self, logits, target):
        n_classes = logits.size(1)
        logits = logits.permute(0, 2, 3, 1).contiguous().view(-1, n_classes)

        target = target.view(-1)
        if self.ignore:
            valid = target != self.ignore
            logits = logits[valid]
            target = target[valid]

        if self.onehot:
            target = to_one_hot(target, n_classes)
        
        probs = F.softmax(logits, dim=1)
        probs = (probs * target).sum(1)
        probs = probs.clamp(self.eps, 1 - self.eps)

        log_p = probs.log()
        batch_loss = -(torch.pow((1 - probs), self.gamma)) * log_p

        if self.size_avg:
            return batch_loss.mean()
        return batch_loss.sum()
