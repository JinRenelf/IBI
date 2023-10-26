# -*- coding: utf-8 -*-

# @File    : IBI_pytorch.py
# @Date    : 2023-09-25
# @Author  : ${ RenJin}


import os
import time
import torch
import pandas as pd
import numpy as np
from datetime import datetime
import scipy.stats as stats
from tqdm import tqdm
import logging

# logging.basicConfig(filename="IBI_pytorch.log",level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logging.getLogger().setLevel(logging.INFO)

def read_variantsF(variants_path_file, variants_size=None):
    """
    read the large genomic file (row_SNPs x column_subjects) using pandas
    if the genomic file is too large,can cover the csv file to pickle file.
    :param variants_path_file:
    :param variants_size: if variants_size=None,select all the variants
    :return: subIDs, varIDs, variants_tensor, df  # list, list, array and dataframe
    """
    if variants_path_file.split(".")[-1] == "csv":
        # this will turn the first column of varIDs into index thus df.columns will only include subIDs
        df = pd.read_csv(variants_path_file, index_col=0)
    else:
        df = pd.read_pickle(variants_path_file)

    if variants_size != None:
        df = df.iloc[:variants_size, :]
    varIDs = list(df.index)
    subIDs = list(int(x) for x in df.columns)
    variants = np.array(df, dtype=np.int8)  # Somehow, np.int8 does not work here.
    A0 = np.ones(len(subIDs), dtype=np.int8)
    variants = np.row_stack((A0, variants))
    varIDs.insert(0, 'A0')
    df = pd.DataFrame(variants, index=varIDs, columns=subIDs, dtype=np.int8)
    variants_tensor = torch.tensor(variants, dtype=torch.float16)
    return subIDs, varIDs, variants_tensor, df

# def read_variantsF(variants_path_file, variants_size=None):
#     """
#     read the large genomic file (row_SNPs x column_subjects) using pandas
#     if the genomic file is too large,can cover the csv file to pickle file.
#     :param variants_path_file:
#     :param variants_size: if variants_size=None,select all the variants
#     :return: subIDs, varIDs, variants_tensor, df  # list, list, array and dataframe
#     """
#     varIDs_file = open(os.path.join(variants_path_file, "varIDs.txt"), "r")
#     varIDs = varIDs_file.read().split("\n")
#     subIDs_file = open(os.path.join(variants_path_file, "subIDs.txt"), "r")
#     subIDs = subIDs_file.read().split("\n")
#     subIDs = list(int(x) for x in subIDs)
#     variants = np.load(os.path.join(variants_path_file, "data.npy"))
#
#     A0 = np.ones(len(subIDs), dtype=np.int8)
#     variants = np.row_stack((A0, variants))
#     varIDs.insert(0, 'A0')
#     # TODO df no use may use too much meomary
#     df = pd.DataFrame()
#     # df = pd.DataFrame(variants, index=varIDs, columns=subIDs, dtype=np.int8)
#     variants_tensor = torch.tensor(variants, dtype=torch.float16)
#
#     if variants_size is not None:
#         varIDs = varIDs[:variants_size+1]
#         variants_tensor = variants_tensor[:variants_size+1,:]
#     return subIDs, varIDs, variants_tensor, df


def read_traitsF(traits_path_file):
    """
    read the .csv file with traits (subjects_row x traits_column)
    :param traits_path_file:
    :return:subIDs, traitIDs, traits_tensor  # list, array,tensor
    """
    # make sure the subIDs become the index column; as default, column names are inferred from the first line of the
    # file
    traits = pd.read_csv(traits_path_file, index_col=0)
    subIDs = list(traits.index)
    traitIDs = traits.columns
    traits_tensor = torch.tensor(traits.values, dtype=torch.float16)
    # np.int8 has changed the type to int8; when using int8, the subIDs become negative.
    #     print(np.sum(traits)) # sum gives odd results of -94.
    return subIDs, traitIDs, traits_tensor  # list, array


def GDsearch_all(traits_tensor, variants_tensor):
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

    # GiVen the Dirichlet Distributions we are using,
    # the expectation of these conditional probabilities is as follows: prior probability
    # P(D=1|V=1) = (alpha11 + V1D1)/(alpha1 + V1D1 + V1D0)*1.0
    cp_D1V1 = (1 + V1D1) / (2 + V1D1 + V1D0) * 1.0
    # P(D=1|V=0) = (alpha01 + V0D1)/(alpha0 + V0D1 + V0D0)*1.0
    cp_D1V0 = (1 + V0D1) / (2 + V0D1 + V0D0) * 1.0
    # RR is risk ratio; OR is oDDs ratio
    RR = cp_D1V1 / cp_D1V0

    # Calculate the log Marginal Likelihood for this particular SNP based on the collected counts and equation 5 in
    # the worD file
    # when j=0 (V=0)
    lgM = torch.lgamma(torch.tensor(2.0)) - torch.lgamma(2.0 + V0D1 + V0D0)
    lgM += torch.lgamma(1.0 + V0D0) - torch.lgamma(torch.tensor(1.0))
    lgM += torch.lgamma(1.0 + V0D1) - torch.lgamma(torch.tensor(1.0))

    # when j=1 (V=1)
    lgM += torch.lgamma(torch.tensor(2.0)) - torch.lgamma(2.0 + V1D1 + V1D0)
    lgM += torch.lgamma(1.0 + V1D0) - torch.lgamma(torch.tensor(1.0))
    lgM += torch.lgamma(1.0 + V1D1) - torch.lgamma(torch.tensor(1.0))

    if variants_tensor.shape[1] == 1:
        # lgM is #traits x 1;otherwise, lgM is, variants x traits.
        lgM = lgM.reshape(1, lgM.shape[0])

    # get the max and index of TopGD across all the rows of variants for each column of the trait inside the 2-D array
    max_value = torch.max(lgM, dim=0).values
    # thus, max_value or max_index is, one vector with the size of K (# of traits)
    max_index = torch.max(lgM, dim=0).indices
    return RR, lgM, max_value, max_index


def lgMcal(variants_tensor, traits_tensor, varID, use_oneTopGD, topGD_index):
    """

    :param varID:
    :param use_oneTopGD:
    :param topGD:
    :return:
    """
    i = varIDs.index(varID)
    # identify the index of patients that have this particular variant Vs=1
    index1 = variants_tensor[i, :] == 1
    index0 = variants_tensor[i, :] == 0

    if use_oneTopGD:
        # we will only consider and search over all the unique topGDs from all the traits;
        V0 = variants_tensor[topGD_index][:, index0]
        # thus one topGD for trait1 may be selected as the sGD for trait2;variants[topGD_index] is mxn
    else:
        # V0 will be [] and its shape will be (0,) if index0 is all false
        V0 = variants_tensor[:, index0]

    # 2478 subjects' hypertension status who have v=1 for this SNP （2478,) and may have HTN=0 or HTN=1
    BP_V1 = traits_tensor[index1]
    # (5290-2478) subjects' hypertension status(1 or 0) who have v=0 for this SNP
    BP_V0 = traits_tensor[index0]
    # print("varID:{}".format(varID))
    lgMv1_SD = DriverSearch(BP_V1, variants_tensor[i, index1])[0]
    # this should be as efficient as SD_lgM_V1; only calculates one marginal assuming SD as the cause, P(D|SD->HT)
    # with [0], the original 2D array, array([[-3127.91831177,...]]),
    # becomes the format of 1D array, array([-3127.91831177,...]),thus consistent with the other output values
    # lgM_v0 is the 2D array; kxk if topGD; m_variants x k_traits if sGD
    lgMv0 = DriverSearch(BP_V0, V0)

    # collect the lgMv0_topGD for each trait in a 1D array; the lgM value for V0 group when using topGD as the driver
    lgMv0_topGD = []
    # collect the r between SD and topGD for each trait in a 1D array
    r = []

    if use_oneTopGD:  # collect the lgMv0_topGD and r for each trait in a 1D array specifically with kxk lgMv0
        for m in range(0, len(traitIDs)):
            lgMv0_topGD.append(lgMv0[m, m])  # with oneTOPGD, lgMv0 is kxk,since k top GD for k traits; here it selects
            # the values of P(D0|topGD-k -> trait-k);
        for j in topGD_index:  # topGD_index is a global variable obtained outside this function
            r1 = stats.spearmanr(variants_tensor[i, :].to("cpu").numpy(), variants_tensor[j, :].to("cpu").numpy())[
                0]
            r.append(r1)
        lgMv0_sGD = torch.zeros(len(traitIDs), device=device)
        sGD = torch.zeros(len(traitIDs), device=device)
    else:
        # with sGD, lgMv0 is m_variants x k_traits
        lgMv0_sGD = torch.max(lgMv0, dim=0).values
        sGD_index = torch.max(lgMv0, dim=0).indices

        sGD = []
        # collect the variant ID of sGD for each trait in a 1D array
        for item in sGD_index:
            sGD.append(varIDs[item])
        sGD = np.array(sGD)

        k = 0
        # collect the lgMv0_topGD and r for each trait in a 1D array specifically with mxk lgMv0
        # topGD_index is one output from GDsearch_all, a vector of K (#traits ordered in the original trait input file)
        for j in topGD_index:
            # a vector of K
            lgMv0_topGD.append(lgMv0[j, k])
            # [0] to get only the coefficient and ignore the p-values
            r1 = stats.spearmanr(variants_tensor[i, :].to("cpu").numpy(), variants_tensor[j, :].to("cpu").numpy())[0]
            r.append(r1)  # a vector of K
            k = k + 1
    lgMv0_topGD = torch.tensor(lgMv0_topGD)
    r = torch.tensor(r)

    if use_oneTopGD:
        lgM_v1v0 = lgMv1_SD + lgMv0_topGD
    else:
        lgM_v1v0 = lgMv1_SD + lgMv0_sGD
    # print(varID,lgMv1_SD,lgMv0_sGD)
    return lgMv1_SD, lgMv0_sGD, lgMv0_topGD, lgM_v1v0, sGD, r, i, varID


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
    # print("VODO:{},V1D0:{},V0D1:{},V1D1:{},lgM:{}".format(V0D0.sum(), V1D0.sum(), V0D1.sum(), V1D1.sum(),lgM))
    return lgM


def save_GDsearch_result(traitIDs, rr, glgm, varIDs, topGD, glgm_topGD):
    """
    collect the headers for the output file
    :param traitIDs:
    :return:
    """
    gstat_head = ['RR', 'M']
    if len(traitIDs) == 1:
        gstat_newhead = gstat_head
    else:
        gstat_newhead = []
        for item in gstat_head:
            for trait in traitIDs:
                new = item + '_' + trait
                gstat_newhead.append(new)
    gstat_newhead.extend(['seq', 'varID'])

    # output the RR and glgm for all the variants
    with open(os.path.join("..", "results", "Ch12wgs_multiTraits_GDsearch_020922_pytorch.csv"),
              "w") as outfile:  # more efficient than using dataframe to_csv...
        outfile.write(','.join(gstat_newhead) + '\n')
        for i in range(0, rr.shape[0]):
            ls = []
            ls.extend(rr[i].tolist())  # row i of rr that is corresponding to the ith variant
            ls.extend(glgm[i].tolist())
            ls.extend([str(i), varIDs[i]])
            outfile.write(','.join(str(item) for item in ls) + '\n')

    with open(os.path.join("..", "results", "Ch12wgs_multiTraits_GDsearch-topGD_020922_pytorch.csv"), "w") as outfile:
        for i in range(0, len(traitIDs)):
            line = [traitIDs[i], str(topGD[i]), str(glgm_topGD[i])]
            #         print(line)
            outfile.write(','.join(str(item) for item in line) + '\n')


def save_sGD_result(element_run):
    """
    collect the headers for this file
    :return:
    """
    # return(lgMv1_SD, lgMv0_sGD, lgMv0_topGD, lgM_v1v0, sGD, r, i, varID)
    outlgM = ['lgMv1_SD', 'lgMv0_sGD', 'lgMv0_topGD', 'lgM_v1v0', 'sGD', 'r']
    if len(traitIDs) == 1:
        outAll = outlgM
    else:
        outAll = []
        for item in outlgM:
            for trait in traitIDs:
                new = item + '_' + trait
                outAll.append(new)
    outAll = outAll + ['seq', 'varID']
    ## element_run is a list; element_run[0] is a tuple of 7 values for one variant from lgMcal function;
    ## element_run[0][0] is the first output of 'lgMv1_SD', a 1D array, array([-3127.91831177,...])
    ### output the big array of element_run (the outputs from lgMcalall_var) to .csv
    with open(os.path.join("", "results", "Ch12wgs_multiTraits_sGD_020522_pytorch.csv"), "w") as outfile:
        # return(lgMv1_SD, lgMv0_sGD, lgMv0_topGD, lgM_v1v0, sGD, i, varID)
        outfile.write(','.join(outAll) + '\n')
        for i in range(0, len(element_run)):  ## Not output 'A0' for easier future analysis?!!
            ls = []
            for j in range(0, len(element_run[0]) - 2):  # the last two elements are not iterable
                # print(element_run[i][j])
                ls.extend(element_run[i][j].tolist())
            ls.extend([element_run[i][-2], element_run[i][-1]])
            # print(ls)
            outfile.write(','.join(str(item) for item in ls) + '\n')


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = "cpu"

    # 1 read data
    start_time = datetime.now()
    # root_path = os.path.join("..","data","1M")
    root_path = os.path.join("..","data","90K")
    # root_path = os.path.join("")
    subIDs, varIDs, variants_tensor, df_variants = read_variantsF(
        # os.path.join(root_path, 'chrm21__KidsFirst_snp01_dominant_withCorrect_Index_RR1.csv'), variants_size=1)
        os.path.join(root_path, 'exonic_variants_01.csv'), variants_size=None)
        # os.path.join(root_path, 'chrm__KidsFirst_snp01_dominant1.csv'), variants_size=None)
        # root_path, variants_size=10)
    # subIDs, varIDs, variants_tensor, df_variants = read_variantsF(
    #     os.path.join(root_path, 'chrm21__KidsFirst_snp01_dominant_withCorrect_Index_RR1.pkl.gzip'), variants_size=100)
    subIDs_BP, traitIDs, traits_tensor = read_traitsF(
        # os.path.join(root_path, 'Phenotype__KidsFirst_withCorrect_Index.csv'))
        os.path.join(root_path, 'Phenotype_exonic_01.csv'))
        # os.path.join(root_path, 'Phenotype__KidsFirst_Index01.csv'))

    end_time = datetime.now()
    elapsed_time = (end_time - start_time).seconds
    logging.info(str(end_time) + '; read data elapsed time: {}s'.format(elapsed_time))
    logging.info("variants: {}".format(np.shape(variants_tensor)))
    logging.info("traits: {}".format(np.shape(traits_tensor)))

    # 2 With GDsearch_all, calculate and output the global stats related to all the traits for all the variants using
    # the entire population cpu better than gpu
    # gpu: cpu:

    variants_tensor = variants_tensor.to(device=device)
    traits_tensor = traits_tensor.to(device=device)
    start_time = datetime.now()
    rr, glgm, glgm_topGD, topGD_index = GDsearch_all(traits_tensor, variants_tensor)
    logging.info("GDsearch all elapsed time: {}s ".format((datetime.now() - start_time).seconds))

    topGD = []
    for item in topGD_index:
        # currently the wgs SNPs are labeled with numbers, thus varIDs and topGD both are int lists.
        topGD.append(varIDs[item])
    # save results
    save_GDsearch_result(traitIDs, rr, glgm, varIDs, topGD, glgm_topGD)

    # 3 sGD search
    #     An important flag to dictate whether using topGD or sGD as the driver for A0 group.
    # use_oneTopGD = False
    # element_run = []
    # start_time = datetime.now()
    # logging.info("device:{}".format(device))
    # try:
    #     for var in tqdm(varIDs):
    #         res = lgMcal(variants_tensor, traits_tensor, var, use_oneTopGD, topGD_index)
    #         element_run.append(res)
    # except Exception as e:
    #     logging.error("Exception occurred", exc_info=True)
    # print("sGD elapsed time: {}s".format((datetime.now() - start_time).seconds))
    #
    # save_sGD_result(element_run)
