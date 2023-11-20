# -*- coding: utf-8 -*-

# @File    : compare_the _performance_of_diff_algorithm.py
# @Date    : 2023-10-27
# @Author  : ${ RenJin}

import os
import torch
import time
import numpy as np
import pandas as pd
from tqdm import tqdm

import logging

# logging.basicConfig(filename="IBI_pytorch.log", level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
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
        variants_tensor, traits_tensor = torch.tensor(variants, dtype=torch.float16, device=devices), \
                                         torch.tensor(traits, dtype=torch.float16, device=devices)
        # return variants_tensor.to_sparse_csr(),traits_tensor.to_sparse_csr()
        return variants_tensor, traits_tensor
    return variants, traits


def data_slicing_method():
    pass


def GDsearch(traits_tensor, variants_tensor):
    """
    Get all the stats for all the variants in any given population for multiple traits;
    particulary used for the entire population;
    Get the nxk matrix of traits==0 and the nxk matrix of traits==1 (n,#subjects;k,#traits)
    if no individuals are in V0 group when the passed variants is [], the V0D0 counts as well as lgM will be 0.

    :param traits_tensor: traits n*k
    :param variants_tensor: variants m*n
    :return:
    """
    bpMask0 = traits_tensor == 0
    d0 = torch.sum(bpMask0)
    bpMask0 = bpMask0.to(torch.float16)

    bpMask1 = traits_tensor == 1
    d1 = torch.sum(bpMask1)
    bpMask1 = bpMask1.to(torch.float16)

    ### Get the mxn vector of snp==0 and the mxn vector of snp==1
    snpMask0 = variants_tensor == 0
    snpMask0 = snpMask0.to(torch.float16)

    snpMask1 = variants_tensor == 1
    snpMask1 = snpMask1.to(torch.float16)

    # Get the four mx1 vector as below: m is # of SNPs in the dataset; for each SNP, the corresponding 4 values
    # from the 4 vectors make up the 2x2 tables between SNP and hypertension
    # Jin V1D1:p(V=1,D=1)calculate the Snp(V)=1 and tarit(D)=1，each Snp's individual num
    V0D0 = snpMask0 @ bpMask0
    V1D0 = snpMask1 @ bpMask0
    V0D1 = snpMask0 @ bpMask1
    V1D1 = snpMask1 @ bpMask1
    return V0D0, V1D0, V0D1, V1D1


def GDsearch_without_data_slicing(traits_tensor, variants_tensor):
    variants_tensor_xor = torch.ones_like(variants_tensor, device=devices) - variants_tensor
    traits_tensor_xor = torch.ones_like(traits_tensor, device=devices) - traits_tensor
    V1D1 = variants_tensor @ traits_tensor
    V1D0 = variants_tensor @ traits_tensor_xor
    V0D1 = variants_tensor_xor @ traits_tensor
    V0D0 = variants_tensor_xor @ traits_tensor_xor
    return V0D0, V1D0, V0D1, V1D1


def GDsearch_without_data_slicing_and_formula_simplification(traits_tensor, variants_tensor):
    variants_sum = variants_tensor.sum(axis=1).reshape(-1, 1)
    traits_sum = traits_tensor.sum(axis=0).reshape(1, -1)
    V1D1 = variants_tensor @ traits_tensor
    V1D0 = torch.ones_like(V1D1) * variants_sum - V1D1
    V0D1 = torch.ones_like(V1D1) * traits_sum - V1D1
    V0D0 = torch.ones_like(V1D0) * variants_tensor.shape[1] - torch.ones_like(V1D0) * traits_sum - V1D0
    return V0D0, V1D0, V0D1, V1D1


class sGDsearch():
    def __init__(self, traits_tensor, variants_tensor):
        self.traits_tensor = traits_tensor
        self.variants_tensor = variants_tensor

    @staticmethod
    def DriverSearch(traits_tensor, variants_tensor):
        bpMask0 = traits_tensor == 0
        bpMask0 = bpMask0.to(torch.float32)
        # 930 HTN and 4360 non-HTN making a totla of 5290 subjects
        d0 = torch.sum(bpMask0)

        bpMask1 = traits_tensor == 1
        bpMask1 = bpMask1.to(torch.float32)
        d1 = torch.sum(bpMask1)

        # Get the mxn vector of snp==0 and the mxn vector of snp==1
        snpMask0 = variants_tensor == 0
        snpMask0 = snpMask0.to(torch.float32)

        snpMask1 = variants_tensor == 1
        snpMask1 = snpMask1.to(torch.float32)

        # Get the four mx1 vector as below: m is # of SNPs in the dataset; for each SNP, the corresponding 4 values
        # from the 4 vectors make up the 2x2 tables between SNP and hypertension
        V0D0 = snpMask0 @ bpMask0  # snpMask0, variants_row x subjects_column
        V1D0 = snpMask1 @ bpMask0  # bpMask0, subjects_row x traits_column
        V0D1 = snpMask0 @ bpMask1
        V1D1 = snpMask1 @ bpMask1
        return V0D0, V1D0, V0D1, V1D1

    def method_inital(self):
        """
         pytorch+loop+data_slicing+mask
        :return:
        """
        for i in tqdm(range(self.variants_tensor.shape[0])):
            # variants=1
            index1 = variants_tensor[i, :] == 1
            V1 = variants_tensor[i, index1]
            BP_V1 = traits_tensor[index1]
            V0D0_1, V1D0_1, V0D1_1, V1D1_1 = self.DriverSearch(BP_V1, V1)
            # variants=0
            index0 = variants_tensor[i, :] == 0
            V0 = variants_tensor[:, index0]
            BP_V0 = traits_tensor[index0]
            V0D0_0, V1D0_0, V0D1_0, V1D1_0 = self.DriverSearch(BP_V0, V0)
            # print(V0D0_1, V1D0_1, V0D1_1, V1D1_1, V0D0_0, V1D0_0, V0D1_0, V1D1_0)

    def method_without_data_slicing(self):
        """
        use matrix multiplication replace data slicing
        :return:
        """
        for i in tqdm(range(self.variants_tensor.shape[0])):
            variants_selected = self.variants_tensor[i, :].reshape(1, -1)
            # variants=1
            weights = variants_selected
            V1 = variants_selected
            V1_xor = (torch.ones(variants_selected.shape, device=devices, dtype=torch.float32) - variants_selected)
            BP_V1 = weights.T * traits_tensor
            BP_V1_1 = (torch.ones(traits_tensor.shape, device=devices, dtype=torch.float32) - traits_tensor) * weights.T
            V1D1_1 = V1 @ BP_V1
            V1D0_1 = V1 @ BP_V1_1
            V0D1_1 = V1_xor @ BP_V1
            V0D0_1 = V1_xor @ BP_V1_1

            # variants=0
            weights = torch.ones((1, variants_tensor.shape[1]), device=devices, dtype=torch.float32) - variants_selected
            variants_tensor_xor = (
                    torch.ones(variants_tensor.shape, device=devices, dtype=torch.float32) - variants_tensor)
            BP_V0 = weights.T * traits_tensor
            BP_V0_1 = (torch.ones(traits_tensor.shape, device=devices, dtype=torch.float32) - traits_tensor) * weights.T

            V1D1_0 = variants_tensor @ BP_V0
            V1D0_0 = variants_tensor @ BP_V0_1
            V0D1_0 = variants_tensor_xor @ BP_V0
            V0D0_0 = variants_tensor_xor @ BP_V0_1
            # print(V0D0_1, V1D0_1, V0D1_1, V1D1_1, V0D0_0, V1D0_0, V0D1_0, V1D1_0)

    def method_without_data_slicing_add_formula_simplification(self):
        for i in tqdm(range(self.variants_tensor.shape[0])):
            variants_selected = self.variants_tensor[i, :].reshape(1, -1)
            # variants=1
            weights = variants_selected
            V1 = variants_selected
            BP_V1 = weights.T * traits_tensor
            BP_V1_sum = BP_V1.sum(axis=0).reshape(1, -1)
            BP_V1_1 = (torch.ones(traits_tensor.shape, device=devices, dtype=torch.float16) - traits_tensor) * weights.T
            BP_V1_1_sum = BP_V1_1.sum(axis=0).reshape(1, -1)
            V1D1_1 = V1 @ BP_V1
            V1D0_1 = V1 @ BP_V1_1
            V0D1_1 = torch.ones_like(V1D1_1) * BP_V1_sum - V1D1_1
            V0D0_1 = torch.ones_like(V1D0_1) * BP_V1_1_sum - V1D0_1

            # variants=0
            weights = torch.ones((1, variants_tensor.shape[1]), device=devices, dtype=torch.float16) - variants_selected
            BP_V0 = weights.T * traits_tensor
            BP_V0_sum = BP_V0.sum(axis=0).reshape(1, -1)
            BP_V0_1 = (torch.ones(traits_tensor.shape, device=devices, dtype=torch.float16) - traits_tensor) * weights.T
            BP_V0_1_sum = BP_V0_1.sum(axis=0).reshape(1, -1)
            V1D1_0 = variants_tensor @ BP_V0
            V1D0_0 = variants_tensor @ BP_V0_1
            V0D1_0 = torch.ones_like(V1D1_0) * BP_V0_sum - V1D1_0
            V0D0_0 = torch.ones_like(V1D0_0) * BP_V0_1_sum - V1D0_0
            # print(V0D0_1, V1D0_1, V0D1_1, V1D1_1, V0D0_0, V1D0_0, V0D1_0, V1D1_0)


if __name__ == '__main__':
    # variants, traits = load_test_demo(variants_num=600000)
    #
    variants = np.array([[0, 1, 1, 0], [0, 0, 1, 0], [0, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 0]])
    traits = np.array([[1], [0], [0], [1]])
    print("variants:", variants.shape)
    # Test the function with sample data 1,000,000*5000: 650s
    devices = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    variants_tensor = torch.tensor(variants, dtype=torch.float16, device=devices)
    traits_tensor = torch.tensor(traits, dtype=torch.float16, device=devices)
    print("devices:", devices)

    """ GD search """
    # # method 1 data slicing
    # start_time = time.time()
    # V0D0_m1, V1D0_m1, V0D1_m1, V1D1_m1 = GDsearch(traits_tensor, variants_tensor)
    # print("method1:all elapsed time:", (time.time() - start_time) * 1000)
    #
    # # method 2 without data slicing
    # torch.cuda.empty_cache()
    # start_time = time.time()
    # V0D0_m2, V1D0_m2, V0D1_m2, V1D1_m2 = GDsearch_without_data_slicing(traits_tensor, variants_tensor)
    # print("method2:all elapsed time:", (time.time() - start_time) * 1000)
    #
    # # method3 without data slicing & formula simplification
    # torch.cuda.empty_cache()
    # start_time = time.time()
    # V0D0_m3, V1D0_m3, V0D1_m3, V1D1_m3 = GDsearch_without_data_slicing_and_formula_simplification(traits_tensor,
    #                                                                                               variants_tensor)
    # print("method3:all elapsed time:", (time.time() - start_time) * 1000)

    """sGDsearch"""
    sGD_search = sGDsearch(traits_tensor, variants_tensor)
    # logging.info("method1: pytorch+loop+data_slicing+mask")
    # sGD_search.method_inital()
    # logging.info("method2: pytorch+loop+matrix_multiplication")
    # sGD_search.method_without_data_slicing()
    logging.info("method3: pytorch+loop+matrix_multiplication+formula_simplification")
    sGD_search.method_without_data_slicing_add_formula_simplification()
