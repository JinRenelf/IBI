# -*- coding: utf-8 -*-

# @File    : csc_matrix.py
# @Date    : 2023-09-25
# @Author  : ${ RenJin}

import os
import time
import scipy
import numpy as np
from scipy import sparse
import pandas as pd
import cupy as cp


def get_col_index(csc):
    csr = csc.tocsr()
    return np.split(csr.indices, csr.indptr[1:-1])[0]


# A = np.array([[0, 1, 1, 0], [1, 0, 0, 0], [0, 0, 1, 1], [1, 0, 0, 0]])
# B = np.array([[0], [1], [0], [0]])
#
df_variants=pd.read_csv(os.path.join("..","data","chrm21__KidsFirst_snp01_dominant_withCorrect_Index_RR1.csv"),index_col=0)
df_traits=pd.read_csv(os.path.join("..","data","Phenotype__KidsFirst_withCorrect_Index.csv"),index_col=0)
A=df_variants.values
B=df_traits.values
cA=cp.asarray(A)
sA = sparse.csr_array(A)
sB = sparse.csr_array(B)

for var in range(1):
    """ 方法一 稀疏矩阵"""
    start_time = time.time()
    variants = A[var, :]
    # Step1: 找到X第一行为0的列索引
    # index1=sA[[var],:].nonzero()[1]
    index1 = get_col_index(sA[[var], :])
    # np.split(index0.indices, index0.indptr[1:-1])
    # Step2: 根据对应的列索引找到traits 和 variants
    BP_V1 = sparse.csr_array(BP_V1)
    V1 = sparse.csr_array(V1)
    # Step3:
    V1D1 = V1 @ BP_V1
    # V1D0 = V1 @ a
    # V1D0 = V1.toarray() @ (1-BP_V1.toarray())

    print("et:", time.time() - start_time)

    """方法二 矩阵相乘"""
    variants = A[var, :]
    start_time = time.time()
    index1 = variants == 1
    BP_V1 = B[index1]
    V1 = A[:, index1]
    V1D1 = V1 @ BP_V1
    # V1D0 = V1 @ (1 - BP_V1)
    print("et:", time.time() - start_time)

    """方法三 现有方法"""
    start_time = time.time()
    index1 = variants == 1
    BP_V1 = B[index1]
    V1 = A[:, index1]
    BP_V0 = np.array(BP_V1==0,dtype=np.int8)
    V0 = np.array(V1==0,dtype=np.int8)
    V1D1 = V1 @ BP_V1
    # V1D0 = V1 @ BP_V0
    print("et:", time.time() - start_time)

    """方法四 cupy"""
    start_time = time.time()
    index1 = variants == 1
    BP_V1 = B[index1]
    V1 = A[:, index1]
    BP_V0 = np.array(BP_V1==0,dtype=np.int8)
    V0 = np.array(V1==0,dtype=np.int8)
    V1D1 = V1 @ BP_V1
    # V1D0 = V1 @ BP_V0
    print("et:", time.time() - start_time)


#
# cA=cp.asarray(A)
# cB=cp.asarray(B)
#
# sRes=sA@sB
# res=sRes.todense()
# cRes=cA@cB
# gamma_res=scipy.special.loggamma(res)

# df_variants=pd.read_csv(os.path.join("..","data","chrm21__KidsFirst_snp01_dominant_withCorrect_Index_RR1.csv"),index_col=0)
# df_traits=pd.read_csv(os.path.join("..","data","Phenotype__KidsFirst_withCorrect_Index.csv"),index_col=0)
# variants=df_variants.values
# traints=df_traits.values
# sVariants=sparse.csc_matrix(variants)
# sTraints=sparse.csc_matrix(traints)
#
# st=time.time()
# res=variants@traints
# et=time.time()
# print("numpy elpased time:",et-st)
#
# st=time.time()
# res=sVariants@sTraints
# et=time.time()
# print("sparse matrix elpased time:",et-st)
#
# for row , col in zip(*sA.nonzero()):
#     print(row,col)
