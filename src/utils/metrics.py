# Copyright (c) 2020 Hai Nguyen
# 
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

import torch 
# import torch.nn.functional as F


smooth = 1.


def dice_coef(logits, labels):
    logits_f = torch.flatten(logits)
    labels_f = torch.flatten(labels)

    intersection = torch.sum(logits_f * labels_f)
    union = torch.sum(logits_f) + torch.sum(labels_f)

    return (2 * intersection + smooth) / (union + smooth)


def true_pos(logits, labels):
    logits_pos = torch.round(logits).type(torch.bool)
    labels_pos = torch.round(labels).type(torch.bool)
    tp = torch.sum(labels_pos * logits_pos) + smooth
    tp_ratio = tp / (torch.sum(labels_pos) + smooth)
    return tp_ratio


def false_pos(logits, labels):
    logits_pos = torch.round(logits).type(torch.bool)
    logits_neg = ~logits_pos
    labels_pos = torch.round(labels).type(torch.bool)
    fp = torch.sum(labels_pos * logits_neg) + smooth
    fp_ratio = fp / (torch.sum(labels_pos) + smooth)
    return fp_ratio


class Metrics(object):
    def __init__(self):
        self.dice = 0
        self.fp = 0
    
    def update(self, preds, labels, i_batch):
        dice = 0
        fp = 0
        b_size = len(labels)
        for lb, pr in zip(labels, preds):
            fp += false_pos(pr, lb)
            dice += dice_coef(pr, lb)
        fp /= b_size
        dice /= b_size
        self.dice = (self.dice * i_batch + dice) / (i_batch + 1)
        self.fp = (self.fp * i_batch + fp) / (i_batch + 1)

    def get_scores(self, phase='train'):
        return {
            f'{phase}_dice': self.dice,
            f'{phase}_fp': self.fp
        }

    def reset(self):
        self.dice = 0
        self.fp = 0
