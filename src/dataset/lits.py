# Copyright (c) 2020 Hai Nguyen
# 
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

import cv2
import numpy as np

import torch
from torch.utils.data import Dataset

from ..utils.transforms import Shift


class LITS(Dataset):
    """
    LIver Tumor Segmentation Dataset (2017)
    """    
    def __init__(
        self,
        slice_paths: list,
        mask_paths: list,
        target_size: tuple = (256, 256),
        seed: int = None,
        p_hflip: float = 0.5,
        p_shift: float = 0.25,
        shift_limit: float = 0.05
    ):
        """LITS Initializer

        Parameters
        ----------
        datadir : str, optional
            [description], by default '.'
        target_size : tuple, optional
            [description], by default (256, 256)
        seed : int, optional
            [description], by default None
        p_hflip : float, optional
            [description], by default 0.5
        p_shift : float, optional
            [description], by default 0.25
        shift_limit : float, optional
            [description], by default 0.05
        """
        super(LITS, self).__init__()

        self.slice_paths = slice_paths
        self.mask_paths = mask_paths
        self.target_size = target_size
        self.p_shift = p_shift

        if seed:
            torch.manual_seed(seed)
            np.random.seed(seed)

        if p_shift:
            self.shift_fn = Shift(limit=shift_limit, prob=p_shift)

    def _transform(self, image: np.ndarray, mask: np.ndarray):
        image = cv2.resize(image, self.target_size)
        mask = cv2.resize(mask, self.target_size)
        image = image.clip(-250, 150)
        mask[mask == mask.max()] = 0
        # image, mask = self.shift_fn()
        return image, mask

    def __getitem__(self, index):
        image = np.load(self.slice_paths[index])
        mask = np.load(self.mask_paths[index])

        image, mask = self._transform(image, mask)

        sample = {
            'image': image,
            'mask': mask,
            'id': index
        }

        return sample

    def __len__(self):
        return len(self.slice_paths)
