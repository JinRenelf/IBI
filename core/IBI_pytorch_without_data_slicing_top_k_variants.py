# -*- coding: utf-8 -*-

# @File    : IBI_pytorch_without_data_slicing_top_k_variants.py
# @Date    : 2023-10-22
# @Author  : ${RenJin}
# -*- coding: utf-8 -*-

# @File    : IBI_pytorch_without_data_slicing.py
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

# logging.basicConfig(filename="IBI_pytorch.log", level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logging.getLogger().setLevel(logging.INFO)


def read_variantsF(variants_path_file):
    """
    read the large genomic file (row_SNPs x column_subjects) using pandas
    but need to convert file to npy
    :param variants_path_file:
    :param variants_size: if variants_size=None,select all the variants
    :return: subIDs, varIDs, variants_tensor, df  # list, list, array and dataframe
    """

    # this will turn the first column of varIDs into index thus df.columns will only include subIDs
    df = pd.read_csv(variants_path_file, index_col=0)

    varIDs = list(df.index)
    subIDs = list(int(x) for x in df.columns)
    variants = np.array(df, dtype=np.int8)  # Somehow, np.int8 does not work here.
    A0 = np.ones(len(subIDs), dtype=np.int8)
    variants = np.row_stack((A0, variants))
    varIDs.insert(0, 'A0')
    # df = pd.DataFrame(variants, index=varIDs, columns=subIDs, dtype=np.int8)
    # when variants data is large, df will consume a large amount of memory,
    # but df is no being used, set df=pd.Dataframe()
    df = pd.DataFrame()
    variants_tensor = torch.tensor(variants, dtype=torch.float32)
    return subIDs, varIDs, variants_tensor, df


def read_variantsF_efficient(variants_path_file):
    """
    read the large genomic file (row_SNPs x column_subjects) using pandas
    if the genomic file is too large,can cover the csv file to pickle file.
    :param variants_path_file:
    :param variants_size: if variants_size=None,select all the variants
    :return: subIDs, varIDs, variants_tensor, df  # list, list, array and dataframe
    """
    varIDs_file = open(os.path.join(variants_path_file, "varIDs.txt"), "r")
    varIDs = varIDs_file.read().split("\n")
    subIDs_file = open(os.path.join(variants_path_file, "subIDs.txt"), "r")
    subIDs = subIDs_file.read().split("\n")
    subIDs = list(int(x) for x in subIDs)
    variants = np.load(os.path.join(variants_path_file, "data.npy"))

    A0 = np.ones(len(subIDs), dtype=np.int8)
    variants = np.row_stack((A0, variants))
    varIDs.insert(0, 'A0')
    # when variants data is large, df will consume a large amount of memory,
    # but df is no being used, set df=pd.Dataframe()
    df = pd.DataFrame()
    # df = pd.DataFrame(variants, index=varIDs, columns=subIDs, dtype=np.int8)
    variants_tensor = torch.as_tensor(variants, dtype=torch.float32)

    return subIDs, varIDs, variants_tensor, df


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
    traits_tensor = torch.as_tensor(traits.values, dtype=torch.float32)
    # np.int8 has changed the type to int8; when using int8, the subIDs become negative.
    #     print(np.sum(traits)) # sum gives odd results of -94.
    return subIDs, traitIDs, traits_tensor  # list, array


def GDsearch_all(traits_tensor, variants_tensor):
    """
    Get all the stats for all the variants in any given population for multiple traits;
    particulary used for the entire population

    :param traits_tensor: traits n*k
    :param variants_tensor: variants m*n
    :return:
    """
    variants_sum = variants_tensor.sum(axis=1).reshape(-1, 1)
    traits_sum = traits_tensor.sum(axis=0).reshape(1, -1)
    V1D1 = variants_tensor @ traits_tensor
    V1D0 = torch.ones_like(V1D1) * variants_sum - V1D1
    V0D1 = torch.ones_like(V1D1) * traits_sum - V1D1
    V0D0 = torch.ones_like(V1D0) * variants_tensor.shape[1] - torch.ones_like(V1D0) * traits_sum - V1D0

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


def cal_lgM(variants_use, weights, traits_tensor, device):
    BP_V = weights.T * traits_tensor
    BP_V_sum = BP_V.sum(axis=0).reshape(1, -1)
    BP_V_xor = (torch.ones(traits_tensor.shape, device=device, dtype=torch.float32) - traits_tensor) * weights.T
    BP_V_xor_sum = BP_V_xor.sum(axis=0).reshape(1, -1)

    V1D1 = variants_use @ BP_V
    V1D0 = variants_use @ BP_V_xor
    V0D1 = torch.ones_like(V1D1) * BP_V_sum - V1D1
    V0D0 = torch.ones_like(V1D0) * BP_V_xor_sum - V1D0

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

    # print("V1D1\n:{},\nV1D0\n:{},\nV0D1\n:{},\nV0D0\n:{},\nlgM:{}".format(V1D1, V1D0, V0D1, V0D0,lgM))
    return lgM


def lgMcal(top_variantsID, top_variants_tensor, variants_tensor, traits_tensor, varID, use_oneTopGD, topGD_index,
           device):
    """

    :param varID:
    :param use_oneTopGD:
    :param topGD:
    :return:
    """
    i = varIDs.index(varID)
    variants_selected = variants_tensor[i, :].reshape(1, -1)
    # when variants=1, calculate lgMv1_SD
    variants_use_1 = variants_selected
    weights_1 = variants_selected
    lgMv1_SD = cal_lgM(variants_use_1, weights_1, traits_tensor, device)

    # when variants=0
    # variants_use_0 = variants_tensor
    variants_use_0 = top_variants_tensor
    weights_0 = torch.ones(variants_selected.shape, dtype=torch.float16, device=device) - variants_selected
    lgMv0 = cal_lgM(variants_use_0, weights_0, traits_tensor, device)

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
        # with sGD, lgMv0 is top_k_variants x k_traits
        lgMv0_sGD = torch.max(lgMv0, dim=0).values
        sGD_index = torch.max(lgMv0, dim=0).indices

        sGD = []
        # collect the variant ID of sGD for each trait in a 1D array
        for item in sGD_index:
            sGD.append(top_variantsID[item])
        sGD = np.array(sGD)

        k = 0
        # collect the lgMv0_topGD and r for each trait in a 1D array specifically with mxk lgMv0
        # topGD_index is one output from GDsearch_all, a vector of K (#traits ordered in the original trait input file)
        for j in topGD_index:
            # a vector of K
            lgMv0_topGD.append(lgMv0[j, k])
            # [0] to get only the coefficient and ignore the p-values
            r1 = stats.spearmanr(variants_tensor[i, :].to("cpu").numpy(), top_variants_tensor[j, :].to("cpu").numpy())[
                0]
            r.append(r1)  # a vector of K
            k = k + 1
    lgMv0_topGD = torch.tensor(lgMv0_topGD)
    r = torch.tensor(r)

    if use_oneTopGD:
        lgM_v1v0 = lgMv1_SD + lgMv0_topGD
    else:
        lgM_v1v0 = lgMv1_SD + lgMv0_sGD
    return lgMv1_SD, lgMv0_sGD, lgMv0_topGD, lgM_v1v0, sGD, r, i, varID


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
    with open(os.path.join("..", "results", "Ch12wgs_multiTraits_GDsearch_020922_pytorch_without_data_slicing.csv"),
              "w") as outfile:  # more efficient than using dataframe to_csv...
        outfile.write(','.join(gstat_newhead) + '\n')
        for i in range(0, rr.shape[0]):
            ls = []
            ls.extend(rr[i].tolist())  # row i of rr that is corresponding to the ith variant
            ls.extend(glgm[i].tolist())
            ls.extend([str(i), varIDs[i]])
            outfile.write(','.join(str(item) for item in ls) + '\n')

    with open(
            os.path.join("..", "results", "Ch12wgs_multiTraits_GDsearch-topGD_020922_pytorch_without_data_slicing.csv"),
            "w") as outfile:
        glgm_topGD = glgm_topGD.cpu().tolist()
        for i in range(0, len(traitIDs)):
            line = [traitIDs[i], str(topGD[i]), str(glgm_topGD[i])]
            #         print(line)
            outfile.write(','.join(str(item) for item in line) + '\n')


def save_sGD_result(element_run, traitIDs, file_name):
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
    with open(os.path.join("..", "results", "{}_sGD_results.csv".format(file_name.split(".")[0])),
              "w") as outfile:
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

# TODO optimize
def GD_search_all_chromosome(chromosome_file_folder_path, phenotype_file_path, device="cpu"):
    subIDs_BP, traitIDs, traits_tensor = read_traitsF(phenotype_file_path)
    traits_tensor = traits_tensor.to(device=device)
    torch.cuda.empty_cache()
    glgm_all = torch.tensor([], device=device)
    varIDs_all = []
    file_list=os.listdir(chromosome_file_folder_path)
    for file_name in file_list:
        subIDs, varIDs, variants_tensor, df_variants = read_variantsF(
            os.path.join(chromosome_file_folder_path, file_name))

        logging.info("variants:{}".format(np.shape(variants_tensor)))
        logging.info("traits: {}".format(np.shape(traits_tensor)))

        # 2 With GDsearch_all, calculate and output the global stats related to all the traits for all the variants
        # move the tensors to GPU
        variants_tensor = variants_tensor.to(device=device)
        torch.cuda.empty_cache()
        start_time = datetime.now()
        rr, glgm, glgm_topGD, topGD_index = GDsearch_all(traits_tensor, variants_tensor)
        logging.info("GDsearch all elapsed time: {}s ".format((datetime.now() - start_time).seconds))
        glgm_all = torch.concat([glgm_all, glgm])
        varIDs_all.extend(varIDs)

    return glgm_all, varIDs_all


def get_top_variants_info(device, k=1000):
    top_vr = pd.read_csv(os.path.join("..", "results", "Top_1000_Vr.csv"), index_col=0)
    top_variantsID = top_vr.index.tolist()
    top_variants = top_vr.values
    A0 = np.ones(top_variants.shape[1], dtype=np.int8)
    top_variants = np.row_stack((A0, top_variants))
    top_variantsID.insert(0, 'A0')
    top_variants_tensor = torch.tensor(top_variants, dtype=torch.float32, device=device)
    return top_variantsID, top_variants_tensor

# only for one trait
def get_top_variants_info_jin(chromosome_file_folder_path, glgm_all, varIDs_all, device, k=1000,union_flag=True):
    """
    select top k variants from all chromosomes, if have multi traits, use union top k variants from all traits
    :param chromosome_file_folder_path:
    :param glgm_all:
    :param varIDs_all:
    :param device:
    :param k:
    :return:
    """
    # Loop through each trait and take the top k variants
    top_k_varIDs_all=[]
    for traits_num in range(glgm_all.shape[1]):
        glgm_all_single=glgm_all[:,traits_num]
        _, sorted_indices = torch.sort(glgm_all_single, descending=True)
        sorted_varIDs_all = [varIDs_all[i] for i in sorted_indices]
        top_k_varIDs = sorted_varIDs_all[:k]
        top_k_varIDs_all.append(top_k_varIDs)
        topGD_varID=sorted_varIDs_all[0]

    if union_flag:
        # take union top k variants from all traits
        top_k_varIDs_union = list(set(x for sublist in top_k_varIDs_all for x in sublist))

    top_k_varID_final = []
    top_k_variants = []
    file_list = os.listdir(chromosome_file_folder_path)

    # extract top k variants from all files
    for file_name in file_list:
        subIDs, varIDs, variants_tensor, df_variants = read_variantsF(
            os.path.join(chromosome_file_folder_path, file_name))
        indexs = [i for i, value in enumerate(varIDs) if value in top_k_varIDs_union]
        selected_varIDs = [value for i, value in enumerate(varIDs) if value in top_k_varIDs_union]
        top_k_varID_final.extend(selected_varIDs)
        top_k_variants.extend(variants_tensor[indexs, :].tolist())

    top_variants=np.array(top_k_variants)
    if "AO" not in top_k_varID_final:
        A0 = np.ones(top_variants.shape[1], dtype=np.int8)
        top_variants = np.row_stack((A0, top_variants))
        top_k_varID_final.insert(0, 'A0')
    top_variants_tensor = torch.tensor(top_variants, dtype=torch.float32, device=device)

    # get topGD_index
    topGD_index=[top_k_varID_final.index(topGD_varID)]
    return top_k_varID_final, top_variants_tensor,topGD_index


if __name__ == '__main__':
    # Create a device with a GPU. If you don't have a GPU, the code will produce an error.
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    chromosome_file_folder_path = os.path.join("..", "data", "all_chromsomes", "CHD All Chrm variant Files")
    phenotype_file_path = os.path.join("..", "data", "all_chromsomes", "CHD Phenotype",
                                       # "Phenotype__KidsFirst_Index01_v4.csv")
                                       "Phenotype_exonic_01.csv")
    k=1000
    glgm_all, varIDs_all= GD_search_all_chromosome(chromosome_file_folder_path, phenotype_file_path, device=device)
    top_variantsID, top_variants_tensor,topGD_index=get_top_variants_info_jin(chromosome_file_folder_path, glgm_all, varIDs_all, device, k=k)

    # 3 sGD search
    # An important flag to dictate whether using topGD or sGD as the driver for A0 group.
    subIDs_BP, traitIDs, traits_tensor = read_traitsF(phenotype_file_path)
    traits_tensor = traits_tensor.to(device=device)
    file_list=os.listdir(chromosome_file_folder_path)

    for file_name in tqdm(file_list):
        subIDs, varIDs, variants_tensor, df_variants = read_variantsF(
            os.path.join(chromosome_file_folder_path, file_name))
        variants_tensor = variants_tensor.to(device=device)

        use_oneTopGD = False
        element_run = []
        start_time = datetime.now()
        logging.info("device:{}".format(device))

        for var in tqdm(varIDs):
            res = lgMcal(top_variantsID, top_variants_tensor, variants_tensor, traits_tensor, var, use_oneTopGD,
                         topGD_index, device=device)
            element_run.append(res)
        save_sGD_result(element_run,traitIDs,file_name)
