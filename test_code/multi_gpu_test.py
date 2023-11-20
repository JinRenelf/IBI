# -*- coding: utf-8 -*-

# @File    : multi_gpu_test.py
# @Date    : 2023-10-10
# @Author  : ${ RenJin}

import torch

import torch

# Assuming you have 2 GPUs
n_gpus = 1

# Create two random large matrices
A = torch.randn(10000, 5000).cuda('cuda:0')
B = torch.randn(5000, 6000).cuda('cuda:0')

# Split matrix A into chunks for each GPU
chunks = torch.chunk(A, n_gpus, dim=0)

# Store results from each multiplication
results = []

for i in range(n_gpus):
    # Send chunk to GPU
    chunk = chunks[i].cuda(f'cuda:{i}')

    # Send matrix B to the same GPU
    B_gpu = B.cuda(f'cuda:{i}')

    # Perform multiplication on the GPU
    result = torch.mm(chunk, B_gpu)

    # Store the result
    results.append(result)

# Concatenate results from all GPUs
# final_result = torch.cat(results, dim=0)


import torch
z = torch.Tensor(([[2,3], [1,1], [4,5]],
                  [[2,2], [1,2], [7,7]],
                  [[2,3], [1,1], [4,5]],
                  [[2,3], [1,1], [4,5]]))

b = torch.Tensor(([1, 0],
                  [0, 1],
                  [0.2, 0.8],
                  [0.5, 0.5]))
print(z.shape, b.shape)

# original implementation
b1 = b.unsqueeze(1)
r1 = z * b1
r1 = torch.sum(r1, dim=-1)
print(r1.shape)

r3 = torch.einsum('ijk,ik->ij', z, b)
print((r1-r3).abs().sum())  # should be zero if we do the same operation