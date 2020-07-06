# Copyright (c) 2020 Hai Nguyen
# 
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

from torchvision import transforms
from .transforms import CLAHE


def get_transforms(input_shape):
    """Get preprocessing pipeline for 
        both images and mask labels

    Parameters
    ----------
    input_shape : list | tuple
        Network's input shape

    Returns
    -------
    tuple
        Including image norm and label norm
    """    
    inorm = transforms.Compose(
        [   
            transforms.Resize((input_shape[0], input_shape[1])),
            transforms.Grayscale(num_output_channels=input_shape[2]),
            CLAHE(),
            transforms.ToTensor(),
        ]
    )

    lnorm = transforms.Compose([
        transforms.Resize((input_shape[0], input_shape[1])),
        transforms.ToTensor()
    ])

    return inorm, lnorm
