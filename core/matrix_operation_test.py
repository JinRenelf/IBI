# -*- coding: utf-8 -*-

# @File    : matrix_operation_test.py
# @Date    : 2023-10-05
# @Author  : ${ RenJin}
import os
import scipy
import torch
import time
import pandas as pd
import numpy as np
from tqdm import tqdm
from joblib import Parallel, delayed


def DriverSearch(traits_tensor, variants_tensor):
    """
    Calcuate and return the lgM for all the drivers or any driver for any given population for multiple traits
    Get the max/min GD and SD as well as their lgM; this can be done in the lgM_cal function so this function can stay the same
    Get the nxk matrix of traits==0 and the nxk matrix of traits==1 (n,#subjects;k,#traits; thus capable of working with multipe traits)
    if no individuals are in V0 group when the passed variants is [], the V0D0 counts as well as lgM will be 0; the max value/index are both turned as 0
    no other SNPs except A0 have a constant value since those have been removed in the preprocessing step;
    :param traits:
    :param variants:
    :return: lgM is a 2D array of #variants x #traits with print(np.shape(lgM))
    """
    bpMask0 = traits_tensor == 0
    bpMask0 = bpMask0.to(torch.float16)
    # 930 HTN and 4360 non-HTN making a totla of 5290 subjects
    d0 = torch.sum(bpMask0)

    bpMask1 = traits_tensor == 1
    bpMask1 = bpMask1.to(torch.float16)
    d1 = torch.sum(bpMask1)

    # Get the mxn vector of snp==0 and the mxn vector of snp==1
    snpMask0 = variants_tensor == 0
    snpMask0 = snpMask0.to(torch.float16)

    snpMask1 = variants_tensor == 1
    snpMask1 = snpMask1.to(torch.float16)

    # Get the four mx1 vector as below: m is # of SNPs in the dataset; for each SNP, the corresponding 4 values
    # from the 4 vectors make up the 2x2 tables between SNP and hypertension
    V0D0 = snpMask0 @ bpMask0  # snpMask0, variants_row x subjects_column
    V1D0 = snpMask1 @ bpMask0  # bpMask0, subjects_row x traits_column
    V0D1 = snpMask0 @ bpMask1
    V1D1 = snpMask1 @ bpMask1
    # print(snpMask1.shape,bpMask0.shape)

    # Calculate the log Marginal LikelihooD for all the SNPs in the matrix based on the collected counts and equation
    # 5 in the worD file when j=0 (V=0)
    lgM = torch.lgamma(torch.tensor(2.0)) - torch.lgamma(2.0 + V0D1 + V0D0)
    lgM += torch.lgamma(1.0 + V0D0) - torch.lgamma(torch.tensor(1.0))
    lgM += torch.lgamma(1.0 + V0D1) - torch.lgamma(torch.tensor(1.0))

    # when j=1 (V=1)
    lgM += torch.lgamma(torch.tensor(2.0)) - torch.lgamma(2.0 + V1D1 + V1D0)
    lgM += torch.lgamma(1.0 + V1D0) - torch.lgamma(torch.tensor(1.0))
    lgM += torch.lgamma(1.0 + V1D1) - torch.lgamma(torch.tensor(1.0))

    if variants_tensor.ndim == 1:
        # lgM is #traits x 1;
        lgM = lgM.reshape(1, lgM.shape[0])

    return lgM, V0D0, V1D0, V0D1, V1D1


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


def scipy_loggmma(V0D1, V0D0, V1D1, V1D0):
    V0D1 = V0D1.cpu().numpy()
    V0D0 = V0D0.cpu().numpy()
    V1D1 = V1D1.cpu().numpy()
    V1D0 = V1D0.cpu().numpy()
    lgM = scipy.special.loggamma(2.0) - scipy.special.loggamma(2.0 + V0D1 + V0D0)
    lgM += scipy.special.loggamma(1.0 + V0D0) - scipy.special.loggamma(1.0)
    lgM += scipy.special.loggamma(1.0 + V0D1) - scipy.special.loggamma(1.0)

    # when j=1 (V=1)
    lgM += scipy.special.loggamma(2.0) - scipy.special.loggamma(2.0 + V1D1 + V1D0)
    lgM += scipy.special.loggamma(1.0 + V1D0) - scipy.special.loggamma(1.0)
    lgM += scipy.special.loggamma(1.0 + V1D1) - scipy.special.loggamma(1.0)
    return lgM


def torch_loggma(V0D1, V0D0, V1D1, V1D0):
    lgM = torch.lgamma(torch.tensor(2.0)) - torch.lgamma(2.0 + V0D1 + V0D0)
    lgM += torch.lgamma(1.0 + V0D0) - torch.lgamma(torch.tensor(1.0))
    lgM += torch.lgamma(1.0 + V0D1) - torch.lgamma(torch.tensor(1.0))

    # when j=1 (V=1)
    lgM += torch.lgamma(torch.tensor(2.0)) - torch.lgamma(2.0 + V1D1 + V1D0)
    lgM += torch.lgamma(1.0 + V1D0) - torch.lgamma(torch.tensor(1.0))
    lgM += torch.lgamma(1.0 + V1D1) - torch.lgamma(torch.tensor(1.0))
    return lgM


if __name__ == '__main__':
    # variants=np.array([[0,1,1,0],[0,0,1,0],[0,0,0,1],[0,1,0,1],[0,0,1,0]])
    # # traits=np.array([[1],[0],[0],[1]])
    # traits=np.array([[1,1],[0,0],[0,0],[1,1]])

    variants, traits = load_test_demo(variants_num=100000)
    # matrix pytorch
    # Test the function with sample data 1,000,000*5000: 650s
    devices = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    variants_tensor = torch.tensor(variants, dtype=torch.float32, device=devices)
    traits_tensor = torch.tensor(traits, dtype=torch.float32, device=devices)

    # search all
    # method2
    start_time = time.time()
    variants_sum = variants_tensor.sum(axis=1).reshape(-1.1)
    traits_sum = traits_tensor.sum(axis=0).reshape(1, -1)
    V1D1_all = variants_tensor @ traits_tensor
    V1D0_all = torch.ones_like(V1D1_all) * variants_sum - V1D1_all
    # V0D1_all = torch.ones_like(V1D1_all) * traits_sum - V1D1_all
    V0D0_all = torch.ones_like(V1D0_all) * variants_tensor.shape[1] - torch.ones_like(V1D0_all) * traits_sum - V1D0_all
    print("method3:all elapsed time:", time.time() - start_time)

    start_time = time.time()
    print("variants: ", np.shape(variants_tensor))
    print("traits: ", np.shape(traits_tensor))
    print("device: ", variants_tensor.device)
    for i in tqdm(range(variants.shape[0])):
        variants_selected = variants_tensor[i, :].reshape(1, -1)
        """variants=1"""
        # 1 method1: pytorch+loop+data_slicing+mask 500k:8min
        index1 = variants_tensor[i, :] == 1
        V1=variants_tensor[i, index1]
        BP_V1 = traits_tensor[index1]
        lgM_m1,V0D0_m1,V1D0_m1,V0D1_m1,V1D1_m1=DriverSearch(BP_V1,V1)

        # 2 method2: pytorch+loop+matrix operation 500k:2min
        weights = variants_selected
        V1=variants_selected
        BP_V1 = weights.T * traits_tensor
        BP_V1_sum = BP_V1.sum(axis=0).reshape(1, -1)
        BP_V1_1 = (torch.ones(traits_tensor.shape,device=devices,dtype=torch.float32) - traits_tensor)* weights.T
        BP_V1_1_sum = BP_V1_1.sum(axis=0).reshape(1, -1)
        V1D1 = V1 @ BP_V1
        V1D0 = V1 @ BP_V1_1
        V0D1 = torch.ones_like(V1D1) * BP_V1_sum - V1D1
        V0D0 = torch.ones_like(V1D0) * BP_V1_1_sum - V1D0

    # print("method1:",i,V1D1_m1,V1D0_m1,V0D1_m1,V0D0_m1)
    # print("method2:",i,V1D1,V1D0,V0D1,V0D0)
    # # check results if equal
    # ##############3 Test start################
    # if not torch.equal(V1D1_m1,V1D1[0]):
    #     print("V1D1 not equal")
    # if not torch.equal(V1D0_m1,V1D0[0]):
    #     print("V1D0 not equal")
    # if not torch.equal(V0D1_m1,V0D1[0]):
    #     print("V0D1 not equal")
    # if not torch.equal(V0D0_m1,V0D0[0]):
    #     print("V0D0 not equal")
    # #############3 Test end ################

    """ variants=0"""
    # method 1 pytorch+loop+data_slicing+mask 100k:GPU 3H  200K 44H
    # index0 = variants_tensor[i, :] == 0
    # V0 = variants_tensor[:, index0]
    # BP_V0 = traits_tensor[index0]
    # lgM,V0D0_m1,V1D0_m1,V0D1_m1,V1D1_m1 =DriverSearch(BP_V0,V0)
    # print(i,V1D1_m1,V1D0_m1,V0D1_m1,V0D0_m1)

    # method 2 pytorch+loop_data_slicing 100k:16min 500K:9H 200K:1H17min 1M:58H
    # weights = torch.ones((1, variants_tensor.shape[1]), device=devices,dtype=torch.float16) - variants_selected
    # BP_V0 = weights.T * traits_tensor
    # BP_V0_sum = BP_V0.sum(axis=0).reshape(1, -1)
    # BP_V0_1 = (torch.ones(traits_tensor.shape,device=devices,dtype=torch.float16) - traits_tensor)* weights.T
    # BP_V0_1_sum = BP_V0_1.sum(axis=0).reshape(1, -1)
    #
    # V1D1 = variants_tensor @ BP_V0
    # V1D0 = variants_tensor @ BP_V0_1
    # V0D1 = torch.ones_like(V1D1) * BP_V0_sum - V1D1
    # V0D0 = torch.ones_like(V1D0) * BP_V0_1_sum - V1D0
    #
    # lgm_scipy=scipy_loggmma(V0D1,V0D0,V1D1,V1D0)
    # lgm_torch=torch_loggma(V0D1,V0D0,V1D1,V1D0)
    # print("scipy:{},torch:{}".format(lgm_scipy,lgm_torch))

    # method 4 Test
    # weights = torch.ones_like(variants_selected, device=devices,dtype=torch.float16) - variants_selected
    # BP_V0 = weights.T * traits_tensor
    # BP_V0_sum = BP_V0.sum(axis=0).reshape(1, -1)
    # BP_V0_1 = (torch.ones(traits_tensor.shape, device=devices,dtype=torch.float16) - traits_tensor) * weights.T
    # BP_V0_1_sum = BP_V0_1.sum(axis=0).reshape(1, -1)
    #
    # BP = torch.hstack((BP_V0, BP_V0_1))
    # V1 = variants_tensor @ BP
    # V1D1 = variants_tensor @ BP_V0
    # V1D0 = variants_tensor @ BP_V0_1
    # V0D1 = BP_V0_sum.repeat(V1.shape[0], 1) - V1[..., :BP_V0.shape[1]]
    # V0D0 = BP_V0_1_sum.repeat(V1.shape[0], 1) - V1[..., :BP_V0_1.shape[1]]

    # ##############3 Test start################
    # if not torch.equal(V1D1_m1,V1D1):
    #     print("not equal")
    # if not torch.equal(V1D0_m1,V1D0):
    #     print("not equal")
    # if not torch.equal(V0D1_m1,V0D1):
    #     print("not equal")
    # if not torch.equal(V0D0_m1,V0D0):
    #     print("not equal")
    # ##############3 Test end ################
    #
    #
    # V1D1_arr=V1D1.cpu().numpy()
    # V1D0_arr=V1D0.cpu().numpy()
    # V0D1_arr=V0D1.cpu().numpy()
    # V0D0_arr=V0D0.cpu().numpy()
    # lgM=torch_loggma(V0D1, V0D0, V1D1, V1D0)
    # lgM_scipy=scipy_loggmma(V0D1_arr, V0D0_arr, V1D1_arr, V1D0_arr)
    # print(lgM,lgM_scipy)
# print("elapsed time:",time.time()-start_time)
# res = lgMcal_no_loop_simplified(variants, traits)
