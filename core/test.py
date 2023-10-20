# -*- coding: utf-8 -*-

# @File    : test.py
# @Date    : 2023-09-20
# @Author  : ${ RenJin}

import time
import scipy
import torch
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("device:",device)

lgM = 0
va = np.array([5])
lgM += scipy.special.loggamma(2.0) - scipy.special.loggamma(2.0 + va)
lgM += scipy.special.loggamma(1.0 + va) - scipy.special.loggamma(1.0)
lgM += scipy.special.loggamma(1.0 + va) - scipy.special.loggamma(1.0)


def test(a):
    res = torch.tensor(np.array([a]), device=device)
    return (res)


res = test(0)

a = torch.tensor([[1, 2, 3], [1, 5, 7]], device=device, dtype=torch.float)
b = torch.tensor([[1, 2], [1, 2], [1, 2]], device=device, dtype=torch.float)
print(torch.mm(a, b))
print(a @ b)


# variants = torch.tensor([[1., 1., 1., 1., 1.],
#                          [0., 0., 0., 0., 0.],
#                          [0., 0., 0., 0., 0.],
#                          [0., 0., 0., 0., 0.],
#                          [0., 0., 0., 0., 0.],
#                          [0., 0., 0., 0., 0.],
#                          [0., 0., 0., 0., 0.],
#                          [0., 0., 0., 0., 0.],
#                          [0., 0., 0., 0., 0.],
#                          [0., 0., 0., 0., 1.],
#                          [0., 0., 0., 0., 0.]], device='cuda:0')
# traits = torch.tensor([[0], [0], [1], [0],[1]], device=device, dtype=torch.float)

# variants = torch.tensor([[0, 1, 1, 0], [0, 0, 1, 0], [0, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 0]], dtype=torch.float,
#                         device='cuda:0')
# traits = torch.tensor([[1], [0], [0], [1]], device=device, dtype=torch.float)

# variants = torch.tensor([[1,1], [0,1],[0,0]], dtype=torch.float,
#                         device='cuda:0')
# traits = torch.tensor([[1], [0]], device=device, dtype=torch.float)
variants_num = 100
individual_num=50
variants = torch.randint(0,2,(variants_num, individual_num),dtype=torch.float16)
traits = torch.randint(0,2,(individual_num, 2),dtype=torch.float16)

"""
for index1 variants=1
for each snp
"""
# t=variants[0,:].reshape(1,-1)
# V1=variants[0,:]*t
# BP_V1=traits*t.T
#
# v1d1=V1@BP_V1
# v1d0=V1@((1-BP_V1)*t.T)
#
# v0d0=((1-V1)*t)@((1-BP_V1)*t.T)
# v0d1=((1-V1)*t)@BP_V1

weights = variants
V1 = variants * weights
BP_V1 = traits * weights.T

print("V1 shape:", V1.shape)
print("BP_V1 shape:", BP_V1.shape)
V1D1 = (V1 * BP_V1.T).sum(axis=1)
V1D0 = (V1 * ((1 - BP_V1) * weights.T).T).sum(axis=1)
V0D0 = (((1 - V1) * weights) * ((1 - BP_V1) * weights.T).T).sum(axis=1)
V0D1 = (((1 - V1) * weights) * BP_V1.T).sum(axis=1)

# m=1000000
# n=5000
# a=np.random.randint(0,2,(m,n))
# b=np.random.randint(0,2,(n,1))
# start_time=time.time()
# res=a@b
# end_time=time.time()
# print("elapsed time:",end_time-start_time)




