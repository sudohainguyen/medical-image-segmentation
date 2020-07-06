# Copyright (c) 2020 Hai Nguyen
# 
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT


def collate(sample):
    image = [b['image'] for b in sample]  # w, h
    label = [b['label'] for b in sample]
    _id = [b["id"] for b in sample]
    return {'images': image, 'labels': label, 'id': _id}


def collate_test(sample):
    image = [b['image'] for b in sample]
    _id = [b["id"] for b in sample]
    return {'images': image, 'id': _id}
