# Copyright (c) 2020 Hai Nguyen
# 
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

import os
import numpy as np

from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision.transforms.functional import hflip

from ..utils.transforms import Shift


class DatasetB(Dataset):
    def __init__(
        self,
        fnames,
        datadir='.',
        target_size=(256, 256),
        image_out_channels=3,
        seed=None,
        hflip=0,
        p_shift=0.25,
        shift_limit=0.05
    ):
        super(DatasetB, self).__init__()
        
        # requires the dataset is organized by 2 sub-folders: 
        # images for image data and masks for groundtruth masks
        self.img_dir = os.path.join(datadir, 'images')
        self.lbl_dir = os.path.join(datadir, 'masks')

        self.fnames = fnames
        self.hflip = hflip
        self.p_shift = p_shift
        if p_shift:
            self.shift_fn = Shift(int(shift_limit * target_size[0]), p_shift)

        self.target_size = target_size

        if seed:
            np.random.seed(seed)
            torch.manual_seed(seed)

    def _transform(self, image, label):
        if self.p_shift:
            image, label = self.shift_fn(image, label)

        if self.hflip and (np.random.random() < self.hflip):
            image = hflip(image)
            label = hflip(label)
        
        return image, label

    def __getitem__(self, index):
        imgpath = os.path.join(self.img_dir, f'{self.fnames[index]}.png')
        lblpath = os.path.join(self.lbl_dir, f'{self.fnames[index]}.png')
        
        image = Image.open(imgpath).convert('RGB')
        label = Image.open(lblpath).convert('L')

        image, label = self._transform(image, label)

        sample = {
            'image': image,
            'label': label,
            'id': self.fnames[index]
        }
        return sample

    def __len__(self):
        return len(self.fnames)
