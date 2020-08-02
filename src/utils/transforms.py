# Copyright (c) 2020 Hai Nguyen
# 
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

import cv2
import numpy as np


class CLAHE:
    def __init__(self, clipLimit=2.0, tileGridSize=(8, 8)):
        self.clipLimit = clipLimit
        self.tileGridSize = tileGridSize

    def __call__(self, im):
        im = np.array(im)
        img_yuv = cv2.cvtColor(im, cv2.COLOR_RGB2YUV)
        clahe = cv2.createCLAHE(clipLimit=self.clipLimit, 
                                tileGridSize=self.tileGridSize)
        img_yuv[:, :, 0] = clahe.apply(img_yuv[:, :, 0])
        img_output = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
        return img_output


class Shift:
    """
    Alternative implementation of shift using opencv
    """    
    def __init__(self, limit=0.2, prob=.5, img_channels=3):
        self.limit = limit
        self.prob = prob
        self.img_channels = img_channels

    def __call__(self, img, mask=None):
        if np.random.random() < self.prob:
            if not isinstance(img, np.ndarray):
                img = np.array(img, dtype=np.uint8)
            height, width = img.shape[:2]
            xlimit = int(self.limit * width)
            ylimit = int(self.limit * height)
            dx = round(np.random.uniform(-xlimit, xlimit))
            dy = round(np.random.uniform(-ylimit, ylimit))
            y1 = ylimit + dy
            y2 = y1 + height
            x1 = xlimit + dx
            x2 = x1 + width

            img1 = cv2.copyMakeBorder(img, ylimit, ylimit, xlimit, xlimit, 
                                      borderType=cv2.BORDER_REFLECT_101)
            if self.img_channels == 1:
                img = img1[y1:y2, x1:x2]
            else:
                img = img1[y1:y2, x1:x2, :]
            if mask is not None:
                if not isinstance(mask, np.ndarray):
                    mask = np.array(mask, dtype=np.uint8)
                msk1 = cv2.copyMakeBorder(mask, ylimit, ylimit, xlimit, xlimit,
                                          borderType=cv2.BORDER_REFLECT_101)
                mask = msk1[y1:y2, x1:x2]
        return img, mask
