# -*- coding: utf-8 -*-

# @File    : test_remove_loop_for_IBI.py
# @Date    : 2023-11-13
# @Author  : ${ RenJin}

import os
import time
import torch
import numpy as np
import pandas as pd
from torch import nn
from tqdm import tqdm
import scipy.stats as stats

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
    """
    Print current memory allocation and peak memory allocation in GB
    :param device:
    :return:
    """
    logging.info("Current GPU Memory allocated: {:.2f} GB".format(torch.cuda.memory_allocated(device) / 1024 ** 3))
    logging.info("Peak GPU Memory allocated: {:.2f} GB".format(torch.cuda.max_memory_allocated(device) / 1024 ** 3))


def cal_lgM(variants_batch, weights_batch, traits, traits_xor, device, variant_type):
    BP_V = torch.einsum('ij,jk->ijk', weights_batch, traits)
    BP_V_xor = torch.einsum('ij,jk->ijk', weights_batch, traits_xor)
    BP_V_sum = BP_V.sum(dim=1).unsqueeze(dim=1)
    BP_V_xor_sum = BP_V_xor.sum(dim=1).unsqueeze(dim=1)
    # variant=1
    if variant_type == 1:
        # V1D1 = torch.einsum('bj,bjk->bk', variants, BP_V)
        # V1D0 = torch.einsum('bj,bjk->bk', variants, BP_V_xor)
        # V0D1 = BP_V_sum - V1D1
        # V0D0 = BP_V_xor_sum - V1D0
        variants_unsqueezed = variants_batch.unsqueeze(dim=1)
        V1D1 = torch.bmm(variants_unsqueezed, BP_V)
        V1D0 = torch.bmm(variants_unsqueezed, BP_V_xor)
        V0D1 = BP_V_sum - V1D1
        V0D0 = BP_V_xor_sum - V1D0
    # variant=0
    else:
        V1D1 = torch.einsum('ij,bjk->bik', variants, BP_V)
        V1D0 = torch.einsum('ij,bjk->bik', variants, BP_V_xor)
        V0D1 = torch.ones_like(V1D1) * BP_V_sum - V1D1
        V0D0 = torch.ones_like(V1D0) * BP_V_xor_sum - V1D0
    # lgM = torch.lgamma(torch.tensor(2.0)) - torch.lgamma(2.0 + V0D1 + V0D0)
    # lgM += torch.lgamma(1.0 + V0D0) - torch.lgamma(torch.tensor(1.0))
    # lgM += torch.lgamma(1.0 + V0D1) - torch.lgamma(torch.tensor(1.0))
    #
    # # when j=1 (V=1)
    # lgM += torch.lgamma(torch.tensor(2.0)) - torch.lgamma(2.0 + V1D1 + V1D0)
    # lgM += torch.lgamma(1.0 + V1D0) - torch.lgamma(torch.tensor(1.0))
    # lgM += torch.lgamma(1.0 + V1D1) - torch.lgamma(torch.tensor(1.0))
    # lgM = lgM.to(torch.float32)

    # if variants_tensor.ndim == 1:
    #     # lgM is #traits x 1;
    #     lgM = lgM.reshape(1, lgM.shape[0])

    # print("V1D1\n:{},\nV1D0\n:{},\nV0D1\n:{},\nV0D0\n:{},\nlgM:{}".format(V1D1, V1D0, V0D1, V0D0,lgM))
    return V1D1, V1D0, V0D1, V0D0


if __name__ == '__main__':
    devices = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    variants, traits = load_test_demo(variants_num=1000000)
    # variants = torch.tensor([[0, 1, 1, 0], [0, 0, 1, 0], [0, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 0]], dtype=torch.float,
    #                         device=devices)
    # varIDs=["SNP1","SNP2","SNP3","SNP4","SNP5"]
    # traits=torch.tensor([[1], [0], [0], [1]],dtype=torch.float,device=devices)
    # traits = torch.tensor([[1, 1], [0, 0], [0, 0], [1, 1]], dtype=torch.float, device=devices)
    variants = torch.tensor(variants, dtype=torch.float32, device=devices)
    traits = torch.tensor(traits, dtype=torch.float32, device=devices)

    get_gpu_meomry_usage(devices)
    logging.info("devices:{}".format(devices))
    logging.info("variants:{}".format(variants.shape))

    use_oneTopGD=False
    topGD_index=[1]

    start_time = time.time()
    # traits_xor = torch.ones(traits.shape, dtype=torch.float16, device=devices) - traits
    batch_size = 512
    results = []
    variants_num = variants.shape[0]
    split_size = int(np.ceil(variants_num / batch_size))
    logging.info("batch_size:{},split_size:{}".format(batch_size, split_size))
    offset = 0
    for i in tqdm(range(split_size)):
        variants_batch = variants[offset:offset + batch_size]
        traits_xor = torch.ones(traits.shape, dtype=torch.float32, device=devices) - traits

        # variant=1
        weights_batch_v1 = variants_batch
        V1D1, V1D0, V0D1, V0D0, lgMv1_SD  = cal_lgM(variants_batch, weights_batch_v1, traits, traits_xor, device=devices,
                                              variant_type=1)
        lgMv1_SD=lgMv1_SD.squeeze(dim=1)

        # variants=0
        weights_batch_v0 = torch.ones(variants_batch.shape, dtype=torch.float32, device=devices) - variants_batch
        V1D1_0, V1D0_0, V0D1_0, V0D0_0, lgMv0 = cal_lgM(variants_batch, weights_batch_v0, traits, traits_xor,
                                                        device=devices, variant_type=0)

        # # collect the lgMv0_topGD for each trait in a 1D array; the lgM value for V0 group when using topGD as the driver
        # lgMv0_topGD = []
        # # collect the r between SD and topGD for each trait in a 1D array
        # r = []
        #
        # if use_oneTopGD:  # collect the lgMv0_topGD and r for each trait in a 1D array specifically with kxk lgMv0
        #     print("need todo")
        #     # for m in range(0, len(traitIDs)):
        #     #     lgMv0_topGD.append(
        #     #         lgMv0[m, m])  # with oneTOPGD, lgMv0 is kxk,since k top GD for k traits; here it selects
        #     #     # the values of P(D0|topGD-k -> trait-k);
        #     # for j in topGD_index:  # topGD_index is a global variable obtained outside this function
        #     #     r1 = stats.spearmanr(variants_tensor[i, :].to("cpu").numpy(), variants_tensor[j, :].to("cpu").numpy())[
        #     #         0]
        #     #     r.append(r1)
        #     # lgMv0_sGD = torch.zeros(len(traitIDs), device=device)
        #     # sGD = torch.zeros(len(traitIDs), device=device)
        # else:
        #     print("doing")
        #     # with sGD, lgMv0 is bath_size*m_variants*k_traits
        #
        #     lgMv0_sGD = torch.max(lgMv0, dim=1).values
        #     sGD_index = torch.max(lgMv0, dim=1).indices
        #
        #     # collect the variant ID of sGD for each trait in a 1D array
        #     sGD = np.array([varIDs[i] for pair in sGD_index for i in pair]).reshape(sGD_index.shape)
        #
        #     k = 0
        #     # collect the lgMv0_topGD and r for each trait in a 1D array specifically with mxk lgMv0
        #     # topGD_index is one output from GDsearch_all, a vector of K (#traits ordered in the original trait input file)
        #     for j in topGD_index:
        #         # a vector of K
        #         lgMv0_topGD.append(lgMv0[:,j, k])
        #         # [0] to get only the coefficient and ignore the p-values
        #         # r1 = stats.spearmanr(variants_batch.to("cpu").numpy(), variants[j, :].to("cpu").numpy())[0]
        #         # r.append(r1)  # a vector of K
        #         k = k + 1
        # # lgMv0_topGD = torch.tensor(lgMv0_topGD)
        #
        # if use_oneTopGD:
        #     lgM_v1v0 = lgMv1_SD + lgMv0_topGD
        # else:
        #     lgM_v1v0 = lgMv1_SD + lgMv0_sGD

        offset += batch_size
    print("elapsed time:", time.time() - start_time)
