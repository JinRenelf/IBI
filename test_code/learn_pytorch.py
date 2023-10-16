# -*- coding: utf-8 -*-

# @File    : learn_pytorch.py
# @Date    : 2023-09-26
# @Author  : ${ RenJin}
import torch
from torch import nn
data = torch.arange(20).reshape(4, 5)
devices = [torch.device('cuda:0'), torch.device('cuda:0')]
split = nn.parallel.scatter(data, devices)
print('input :', data)
print('load into', devices)
print('output:', split)
