# -*- coding: utf-8 -*-

# @File    : multi_gpu_test.py
# @Date    : 2023-10-10
# @Author  : ${ RenJin}

import torch

import torch

# Assuming you have 2 GPUs
n_gpus = 2

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
final_result = torch.cat(results, dim=0)