# -*- coding: utf-8 -*-

# @File    : learn_pytorch.py
# @Date    : 2023-09-26
# @Author  : ${ RenJin}
import os
import time
import torch
import numpy as np
import pandas as pd
from torch import nn

def load_test_demo(variants_num = None,isSparse=False,toTensor=False):
    variants=np.load(os.path.join("..", "data", "1M", "data.npy"))
    if variants_num!=None:
        variants=variants[:variants_num,:]
    traits=pd.read_csv(os.path.join("..", "data", "1M", "Phenotype__KidsFirst_Index01.csv"), index_col=0)
    traits=traits.values

    if isSparse and not toTensor:
        from scipy import sparse
        variants_sparse, traits_sparse = sparse.csr_matrix(variants), sparse.csr_matrix(traits)
        return variants_sparse, traits_sparse
    elif toTensor:
        devices = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        variants_tensor, traits_tensor = torch.tensor(variants,dtype=torch.float32,device=devices), \
                                         torch.tensor(traits,dtype=torch.float32,device=devices)
        # return variants_tensor.to_sparse_csr(),traits_tensor.to_sparse_csr()
        return variants_tensor, traits_tensor
    return variants,traits

if __name__ == '__main__':

    # variants=torch.tensor([[0,1,1,0],[0,0,1,0],[0,0,0,1],[0,1,0,1],[0,0,1,0]],dtype=torch.float)
    # traits=torch.tensor([[1], [0], [0], [1]],dtype=torch.float)

    variants, traits = load_test_demo(variants_num=100000)
    devices = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    variants=torch.tensor(variants,dtype=torch.float16,device=devices)
    traits=torch.tensor(traits,dtype=torch.float16,device=devices)


    start_time=time.time()
    traits_xor=torch.ones(traits.shape,dtype=torch.float16,device=devices)-traits
    # Repeat the tensor to get 40 elements
    nvariants=variants.shape[0]
    repeated_traits = traits.repeat(nvariants, 1).reshape(nvariants, traits.shape[0],traits.shape[1])
    repeated_traits_xor = traits_xor.repeat(nvariants, 1).reshape(nvariants, traits_xor.shape[0],traits_xor.shape[1])

    # Expand dimensions of 'a' to make it compatible for element-wise multiplication
    expanded_a = variants.unsqueeze(2).expand_as(repeated_traits)

    # Perform element-wise multiplication
    repeated_traits_with_weight = expanded_a * repeated_traits
    repeated_traits_with_weight_xor=expanded_a * repeated_traits_xor
    # Adjust dimensions for matrix multiplication
    expanded_variants_for_matmul = expanded_a.permute(0, 2, 1)
    #
    V1D1=torch.bmm(expanded_variants_for_matmul, repeated_traits_with_weight)
    V1D0=torch.bmm(expanded_variants_for_matmul,repeated_traits_with_weight_xor)
    print("elapsed time:",time.time()-start_time)
    # matmul_results = torch.bmm(expanded_a, repeated_traits_with_weight)
    #
    # a = torch.tensor([[0,1,1,0],[0,0,1,0],[0,0,0,1],[0,1,0,1],[0,0,1,0]])
    # # b = torch.tensor([[1, 1], [0, 0], [0, 0], [1, 1]])
    # b = torch.tensor([[1], [0], [0], [1]])
    #
    # # Repeat the tensor to get 40 elements
    # repeated_b = b.repeat(5, 1).view(5, 4, 1)
    #
    # # Expand dimensions of 'a' to make it compatible for element-wise multiplication
    # expanded_a = a.unsqueeze(2).expand_as(repeated_b)
    #
    # # Perform element-wise multiplication
    # result = expanded_a * repeated_b
    #
    # # Adjust dimensions for matrix multiplication
    # expanded_a_for_matmul = expanded_a.permute(0, 2, 1)  # shape becomes (5, 2, 4)
    #
    # # Perform batched matrix multiplication
    # matmul_results = torch.bmm(expanded_a_for_matmul, result)
    #
    # print(matmul_results)
    #
    #
