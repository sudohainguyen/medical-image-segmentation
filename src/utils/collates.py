# Copyright (c) 2020 Hai Nguyen
# 
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

import torch


def collate(sample):
    image = torch.stack([b['image'] for b in sample], dim=0)  # w, h
    mask = torch.stack([b['mask'] for b in sample], dim=0)
    _id = [b["id"] for b in sample]
    return {'images': image, 'masks': mask, 'id': _id}


def collate_test(sample):
    image = torch.stack([b['image'] for b in sample], dim=0)
    _id = [b["id"] for b in sample]
    return {'images': image, 'id': _id}
