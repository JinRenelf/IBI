# -*- coding: utf-8 -*-

# @File    : pytorch_multi_gpu.py
# @Date    : 2023-09-26
# @Author  : ${ RenJin}
import os
import time
import torch
import numpy as np
import pandas as pd
from torch import nn
from tqdm import tqdm

import logging

logging.getLogger().setLevel(logging.INFO)


def load_test_demo(variants_num=None, isSparse=False, toTensor=False):
    variants = np.load(os.path.join("..", "data", "1M", "1M_variants_value.npy"))
    if variants_num != None:
        variants = variants[:variants_num, :]
    traits = pd.read_csv(os.path.join("..", "data", "1M", "Phenotype__KidsFirst_Index01.csv"), index_col=0)
    traits = traits.values

    if isSparse and not toTensor:
        from scipy import sparse
        variants_sparse, traits_sparse = sparse.csr_matrix(variants), sparse.csr_matrix(traits)
        return variants_sparse, traits_sparse
    elif toTensor:
        devices = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        variants_tensor, traits_tensor = torch.tensor(variants, dtype=torch.float32, device=devices), \
                                         torch.tensor(traits, dtype=torch.float32, device=devices)
        # return variants_tensor.to_sparse_csr(),traits_tensor.to_sparse_csr()
        return variants_tensor, traits_tensor
    return variants, traits


def get_gpu_meomry_usage(device):
    # Print current memory allocation and peak memory allocation in GB
    logging.info("Current GPU Memory allocated: {:.2f} GB".format(torch.cuda.memory_allocated(device) / 1024 ** 3))
    logging.info("Peak GPU Memory allocated: {:.2f} GB".format(torch.cuda.max_memory_allocated(device) / 1024 ** 3))


def get_split_batch(batch_size, traits_unsqueezed):
    variants_num = traits_unsqueezed.shape[0]
    split_size = int(np.ceil(variants_num / batch_size))
    offset = 0
    split_batch = []
    for i, device in enumerate(range(split_size)):
        split = traits_unsqueezed[offset:offset + batch_size]
        split_batch.append(split)
        offset += batch_size
    return split_batch


if __name__ == '__main__':
    variants, traits = load_test_demo(variants_num=100000)
    #
    devices = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # variants=torch.tensor([[0,1,1,0],[0,0,1,0],[0,0,0,1],[0,1,0,1],[0,0,1,0]],dtype=torch.float,device=devices)
    # traits=torch.tensor([[1], [0], [0], [1]],dtype=torch.float,device=devices)
    # traits=torch.tensor([[1,1], [0,0], [0,0], [1,1]],dtype=torch.float,device=devices)

    # variants=torch.tensor([[0,1],[1,0],[0,0]],dtype=torch.float,device=devices)
    # traits=torch.tensor([[1], [0]],dtype=torch.float,device=devices)
    variants = torch.tensor(variants, dtype=torch.float16, device=devices)
    traits = torch.tensor(traits, dtype=torch.float16, device=devices)

    get_gpu_meomry_usage(devices)

    logging.info("devices:{}".format(devices))
    logging.info("variants:{}".format(variants.shape))

    start_time = time.time()
    traits_xor = torch.ones(traits.shape, dtype=torch.float16, device=devices) - traits
    # BP_V1 = torch.einsum('ij,jk->ijk', variants, traits)
    # BP_V1_xor = torch.einsum('ij,jk->ijk', variants, traits_xor)
    #
    # V1D1 = torch.einsum('ij,ijk->ik', variants, BP_V1)
    # V1D0 = torch.einsum('ij,ijk->ik', variants, BP_V1_xor)

    # Adjust dimensions for matrix multiplication
    # Unsqueeze variants to add a broadcasting dimension: from (5, 4) to (1, 5, 4)
    # variants_unsqueezed = variants.unsqueeze(dim=1)
    # V1D1 = torch.bmm(variants_unsqueezed, BP_V1)
    # V1D0 = torch.bmm(variants_unsqueezed, BP_V1_xor)
    # BP_V1_sum = BP_V1.sum(dim=1)
    # BP_V1_xor_sum = BP_V1_xor.sum(dim=1)
    # V0D1 = BP_V1_sum.unsqueeze(dim=1) - V1D1
    # V0D0 = BP_V1_xor_sum.unsqueeze(dim=1) - V1D0

    BP_V1 = torch.einsum('ij,jk->ijk', variants, traits)
    BP_V1_xor = torch.einsum('ij,jk->ijk', variants, traits_xor)

    st = time.time()
    # Adjust dimensions for matrix multiplication
    # Unsqueeze variants to add a broadcasting dimension: from (5, 4) to (1, 5, 4)
    variants_unsqueezed = variants.unsqueeze(dim=1)
    start_time=time.time()
    V1D1 = torch.bmm(variants_unsqueezed, BP_V1 )
    print("11elapsed time:",time.time()-start_time)

    start_time=time.time()
    # q=1
    V1D1_1=torch.einsum("mqn,mnk->mqk", variants_unsqueezed, BP_V1)
    print("22elapsed time:",time.time()-start_time)
    V1D0 = torch.bmm(variants_unsqueezed, BP_V1_xor)
    V1D0_1 = torch.einsum("mqn,mnk->mqk", variants_unsqueezed, BP_V1_xor)
    BP_V1_sum = BP_V1.sum(dim=1)
    BP_V1_xor_sum = BP_V1_xor.sum(dim=1)
    V0D1 = BP_V1_sum.unsqueeze(dim=1) - V1D1
    V0D0 = BP_V1_xor_sum.unsqueeze(dim=1) - V1D0

    logging.info("elapsed time:{}".format(time.time() - start_time))
    get_gpu_meomry_usage(devices)
    # del BP_V1, BP_V1_xor, variants_unsqueezed, BP_V1_xor_sum, BP_V1_sum
    torch.cuda.empty_cache()
    get_gpu_meomry_usage(devices)

    #
    logging.info("elapsed time:{}".format(time.time() - start_time))
    # get_gpu_meomry_usage(devices)
    # del repeated_traits_with_weight, repeated_traits_with_weight_xor, variants_unsqueezed, BP_V1_xor_sum, BP_V1_sum
    torch.cuda.empty_cache()
    get_gpu_meomry_usage(devices)

    # variants=0
    st = time.time()
    get_gpu_meomry_usage(devices)

    batch_size = 512
    results = []
    variants_num = variants.shape[0]
    split_size = int(np.ceil(variants_num / batch_size))
    logging.info("batch_size:{},split_size:{}".format(batch_size, split_size))

    offset = 0
    for i in tqdm(range(split_size)):
        variants_batch = variants[offset:offset + batch_size]
        traits_xor = torch.ones(traits.shape, dtype=torch.float16, device=devices) - traits
        weights_batch = torch.ones(variants_batch.shape, dtype=torch.float16, device=devices) - variants_batch

        BP_V0 = torch.einsum('ij,jk->ijk', weights_batch, traits)
        BP_V0_xor=torch.einsum('ij,jk->ijk',weights_batch,traits_xor)
        V1D1_0 = torch.einsum('ij,bjk->bik', variants, BP_V0)
        V1D0_0 = torch.einsum('ij,bjk->bik', variants,BP_V0_xor)

        BP_V0_sum = BP_V0.sum(dim=1).unsqueeze(dim=1)
        BP_V0_xor_sum = BP_V0_xor.sum(dim=1).unsqueeze(dim=1)
        V0D1_0 = torch.ones_like(V1D1_0) * BP_V0_sum - V1D1_0
        V0D0_0 = torch.ones_like(V1D0_0) * BP_V0_xor_sum - V1D0_0
        offset+=batch_size

    # # repeated_traits_with_weight_xor_unsqueezed  = torch.einsum('ij,jk->ijk', weights, traits_xor).unsqueeze(1)
    # # #
    # # variants_unsqueezed = variants.unsqueeze(0)
    # # del weights
    # # torch.cuda.empty_cache()
    # # # Unsqueeze traits in the second dimension: from (5, 4, 1) to (5, 1, 4)
    # # # repeated_traits_with_weight_unsqueezed = repeated_traits_with_weight.unsqueeze(1)
    # # # repeated_traits_with_weight_xor_unsqueezed = repeated_traits_with_weight_xor.unsqueeze(1)
    # # logging.info("get repeated traits")
    # # get_gpu_meomry_usage(devices)
    # # # Perform batch matrix multiplication using torch.matmul
    # # # The shapes are now variants_unsqueezed: (1, 5, 4), traits_unsqueezed: (5, 1, 4)
    # # # PyTorch will broadcast the tensors to a compatible size during multiplication
    # # batch_size = 1000
    # #
    # # results = []
    # #
    # # variants_num = variants.shape[0]
    # # split_size = int(np.ceil(variants_num / batch_size))
    # # logging.info("batch_size:{},split_size:{}".format(batch_size,split_size))
    # # offset = 0
    # #
    # # for i in tqdm(range(split_size)):
    # #     # get_gpu_meomry_usage(devices)
    # #     traits_batch = repeated_traits_with_weight_unsqueezed[offset:offset + batch_size]
    # #     traits_xor_batch = repeated_traits_with_weight_xor_unsqueezed[offset:offset + batch_size]
    # #     V1D1 = torch.matmul(variants_unsqueezed, traits_batch).squeeze(1)
    # #     V1D0 = torch.matmul(variants_unsqueezed, traits_xor_batch).squeeze(1)
    # #
    # #     BP_V1_sum = traits_batch.sum(dim=2)
    # #     BP_V1_xor_sum = traits_xor_batch.sum(dim=2)
    # #     V0D1 = torch.ones_like(V1D1)*BP_V1_sum - V1D1
    # #     V0D0 = torch.ones_like(V1D0)*BP_V1_xor_sum - V1D0
    # #
    # #     del traits_batch, traits_xor_batch
    # #     torch.cuda.empty_cache()
    # #
    # #     # results.extend(result)
    # #
    # print("elapsed time:", time.time() - st)
