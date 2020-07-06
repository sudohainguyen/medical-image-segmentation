# Copyright (c) 2020 Hai Nguyen
# 
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

from termcolor import cprint
import torch
import torch.nn as nn


def get_activation_layer(name, inplace=True) -> nn.Module:
    """Get activation layer by name

    Parameters
    ----------
    name : str
        Activation function name
    inplace : bool, optional
        Inplace, by default True

    Returns
    -------
    nn.Module
    """        
    activation = {
        'relu': nn.ReLU(inplace=inplace),
        'relu6': nn.ReLU6(inplace=inplace),
        'sigmoid': nn.Sigmoid(),
        'softmax': nn.Softmax()
    }
    return activation[name]


def init_weights(net, init_type='xavier', gain=0.02): 
    """Weight initialization strategy for nn layers

    Parameters
    ----------
    net : nn.Module
        NN to apply initialization
    init_type : str, optional
        Initialization type, by default 'xavier'
    gain : float, optional
        Scaling factor, by default 0.02
    """

    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):  # noqa: E501
            if init_type == 'normal':
                nn.init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                nn.init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                nn.init.orthogonal_(m.weight.data, gain=gain)
            else:
                msg = f'initialization method {init_type} is not implemented'
                raise NotImplementedError(msg)
            
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            nn.init.normal_(m.weight.data, 1.0, gain)
            nn.init.constant_(m.bias.data, 0.0)
    net.apply(init_func)


def to_one_hot(y, n_dims=None):
    """Convert label tensors with n dims to one-hot representation
        with n+1 dims.

    Parameters
    ----------
    y : Tensor, Variable
        Input tensor    
    n_dims : int, optional
        Conveted tensor's number of dims, by default None

    Returns
    -------
    Tensor
        One-hot tensor
    """    

    y_tensor = y.data if isinstance(y, torch.Variable) else y
    y_tensor = y_tensor.type(torch.LongTensor).view(-1, 1)
    n_dims = n_dims if n_dims is not None else int(torch.max(y_tensor)) + 1
    
    y_one_hot = torch.zeros(y_tensor.size()[0], n_dims) \
        .scatter_(1, y_tensor, 1)
    y_one_hot = y_one_hot.view(*y.shape, -1)

    return torch.Variable(y_one_hot) \
        if isinstance(y, torch.Variable) else y_one_hot


def one_hot(labels, num_classes: int, dtype: torch.dtype = torch.float):
    """
    For a tensor `labels` of dimensions B1[spatial_dims], 
    return a tensor of dimensions `BN[spatial_dims]`
    for `num_classes` N number of classes.

    Example:

        For every value v = labels[b,1,h,w], the value in the result at 
            [b,v,h,w] will be 1 and all others 0.
        Note that this will include the background label, 
            thus a binary mask should be treated as having 2 classes.
    """
    assert labels.dim() > 0, "labels should have dim of 1 or more."

    # if 1D, add singelton dim at the end
    if labels.dim() == 1:
        labels = labels.view(-1, 1)

    sh = list(labels.shape)

    assert sh[1] == 1, "labels should have a channel \
        with length equals to one."
    sh[1] = num_classes

    o = torch.zeros(size=sh, dtype=dtype, device=labels.device)
    labels = o.scatter_(dim=1, index=labels.long(), value=1)

    return labels


def print_info(info, _type=None):
    if _type is not None:
        if isinstance(info, str):
            cprint(info, _type[0], attrs=[_type[1]])
        elif isinstance(info, list):
            for i in range(info):
                cprint(i, _type[0], attrs=[_type[1]])
    else:
        print(info)
