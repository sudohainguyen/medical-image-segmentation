# Copyright (c) 2020 Hai Nguyen
# 
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

import os
import numpy as np

from PIL import Image

from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms.functional import hflip, to_pil_image

from ..utils.transforms import Shift, CLAHE


class DatasetB(Dataset):
    """
    Dataset B for Breast Cancer ultrasound imaging.
    """
    def __init__(
        self,
        fnames: list,
        datadir: str = '.',
        target_size: tuple = (256, 256),
        image_out_channels: int = 3,
        p_hflip: float = 0.5,
        p_shift: float = 0.25,
        shift_limit: float = 0.05
    ):
        """Dataset B initializer

        Parameters
        ----------
        fnames : list
            List of image file name without extension part
        datadir : str, optional
            Data directory, by default '.'
        target_size : tuple, optional
            Images will be resized to this shape, by default (256, 256)
        image_out_channels : int, optional
            Number channels of output images could be either 3 or 1,
            by default 3
        seed : int, optional
            Random seed for augmentation, by default None
        hflip : float, optional
            Horizontaly flip random rate, by default 0
        p_shift : float, optional
            Shift rates, by default 0.25
        shift_limit : float, optional
            Proportion of maximum distance comparing with image size that 
                will be used for shifting,
                by default 0.05
        """        
        super(DatasetB, self).__init__()
        
        # requires the dataset is organized by 2 sub-folders: 
        # images for image data and masks for groundtruth masks
        self.img_dir = os.path.join(datadir, 'images')
        self.lbl_dir = os.path.join(datadir, 'masks')

        self.fnames = fnames
        self.p_hflip = p_hflip
        self.p_shift = p_shift
        if p_shift:
            self.shift_fn = Shift(int(shift_limit * target_size[0]), p_shift)

        self.inorm = transforms.Compose(
            [   
                transforms.Resize(target_size),
                CLAHE(),
                transforms.ToPILImage(),
                transforms.Grayscale(num_output_channels=image_out_channels),
                transforms.ToTensor(),
            ]
        )
        self.lnorm = transforms.Compose([
            transforms.Resize(target_size),
            transforms.ToTensor()
        ])

    def _augment(self, image, mask):
        if self.p_shift:
            image, mask = self.shift_fn(image, mask)
            if not isinstance(image, Image.Image):
                image = to_pil_image(image)
                mask = to_pil_image(mask)

        if self.p_hflip and (np.random.random() < self.p_hflip):
            image = hflip(image)
            mask = hflip(mask)
        return image, mask

    def _preprocess(self, image, mask):
        return self.inorm(image), self.lnorm(mask)

    def __getitem__(self, index):
        imgpath = os.path.join(self.img_dir, f'{self.fnames[index]}.png')
        lblpath = os.path.join(self.lbl_dir, f'{self.fnames[index]}.png')
        
        image = Image.open(imgpath).convert('RGB')
        mask = Image.open(lblpath).convert('L')

        image, mask = self._augment(image, mask)
        image, mask = self._preprocess(image, mask)

        sample = {
            'image': image,
            'mask': mask,
            'id': self.fnames[index]
        }
        return sample

    def __len__(self):
        return len(self.fnames)
