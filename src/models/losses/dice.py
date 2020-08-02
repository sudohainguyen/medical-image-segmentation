# Copyright (c) 2020 Hai Nguyen
# 
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

import warnings

import torch
from torch.nn.modules.loss import _Loss

from ...utils.funcs import one_hot


class DiceLoss(_Loss):

    def __init__(
        self,
        include_background: bool = True,
        to_onehot_y: bool = False,
        sigmoid: bool = False,
        softmax: bool = False,
        squared_pred: bool = False,
        jaccard: bool = False,
        reduction: str = "mean",
    ):
        """
        Args:
            include_background: If False channel index 0 (background category) 
                is excluded from the calculation.
            to_onehot_y: whether to convert `y` into the one-hot format. 
                Defaults to False.
            sigmoid: If True, apply a sigmoid function to the prediction.
            softmax: If True, apply a softmax function to the prediction.
            squared_pred: use squared versions of targets and predictions 
                in the denominator or not.
            jaccard: compute Jaccard Index (soft IoU) instead of dice or not.
            reduction (`none|mean|sum`): Specifies the reduction to apply to 
                the output:
                ``'none'``: no reduction will be applied,
                ``'mean'``: the sum of the output will be divided by 
                    the number of elements in the output,
                ``'sum'``: the output will be summed.
                Default: ``'mean'``.
        """
        super().__init__(reduction=reduction)

        if reduction not in ["none", "mean", "sum"]:
            raise ValueError(f"reduction={reduction} is invalid. \
                Valid options are: none, mean or sum.")

        if sigmoid and softmax:
            raise ValueError("sigmoid=True and softmax=True \
                are not compatible.")

        self.include_background = include_background
        self.to_onehot_y = to_onehot_y
        self.sigmoid = sigmoid
        self.softmax = softmax
        self.squared_pred = squared_pred
        self.jaccard = jaccard

    def forward(self, input: torch.Tensor, 
                target: torch.Tensor, smooth: float = 1e-5):
        """
        Args:
            input (tensor): the shape should be BNH[WD].
            target (tensor): the shape should be BNH[WD].
            smooth: a small constant to avoid nan.
        """
        if self.sigmoid:
            input = torch.sigmoid(input)

        n_pred_ch = input.shape[1]
        if n_pred_ch == 1:
            if self.softmax:
                warnings.warn("single channel prediction, \
                    `softmax=True` ignored.")
            if self.to_onehot_y:
                warnings.warn("single channel prediction, \
                    `to_onehot_y=True` ignored.")
            if not self.include_background:
                warnings.warn("single channel prediction, \
                    `include_background=False` ignored.")
        else:
            if self.softmax:
                input = torch.softmax(input, 1)

            if self.to_onehot_y:
                target = one_hot(target, num_classes=n_pred_ch)
            if not self.include_background:
                # if skipping background, removing first channel
                target = target[:, 1:]
                input = input[:, 1:]

        reduce_axis = list(range(2, len(input.shape)))
        intersection = torch.sum(target * input, dim=reduce_axis)

        if self.squared_pred:
            target = torch.pow(target, 2)
            input = torch.pow(input, 2)

        ground_o = torch.sum(target, dim=reduce_axis)
        pred_o = torch.sum(input, dim=reduce_axis)

        denominator = ground_o + pred_o

        if self.jaccard:
            denominator -= intersection

        f = 1.0 - (2.0 * intersection + smooth) / (denominator + smooth)

        if self.reduction == "mean":
            f = torch.mean(f)  # the batch and channel average
        elif self.reduction == "sum":
            f = torch.sum(f)  # sum over the batch and channel dims
        elif self.reduction == "none":
            pass  # returns [N, n_classes] losses
        else:
            raise ValueError(f"reduction={self.reduction} is invalid.")

        return f
