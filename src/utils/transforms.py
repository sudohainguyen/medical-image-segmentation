# Copyright (c) 2020 Hai Nguyen
# 
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

import numpy as np
import cv2
from PIL import Image


class CLAHE:
    """
    Contrast-limited adaptive histogram equalization
    """
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
    def __init__(self, limit=4, prob=.5):
        self.limit = limit
        self.prob = prob

    def __call__(self, img, mask=None):
        if np.random.random() < self.prob:
            img = np.array(img, dtype=np.uint8)
            limit = self.limit
            dx = round(np.random.uniform(-limit, limit))
            dy = round(np.random.uniform(-limit, limit))
            height, width, _ = img.shape
            y1 = limit + dy
            y2 = y1 + height
            x1 = limit + dx
            x2 = x1 + width

            img1 = cv2.copyMakeBorder(img, limit, limit, limit, limit, 
                                      borderType=cv2.BORDER_REFLECT_101)
            img = img1[y1:y2, x1:x2, :]
            img = Image.fromarray(img)
            if mask is not None:
                mask = np.array(mask, dtype=np.uint8)
                msk1 = cv2.copyMakeBorder(mask, limit, limit, limit, limit,
                                          borderType=cv2.BORDER_REFLECT_101)
                mask = msk1[y1:y2, x1:x2]
                mask = Image.fromarray(mask)
        return img, mask
