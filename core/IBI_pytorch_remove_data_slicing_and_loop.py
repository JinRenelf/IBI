# -*- coding: utf-8 -*-

# @File    : IBI_pytorch_remove_data_slicing_and_loop.py
# @Date    : 2023-11-15
# @Author  : RenJin

import os
import time
import torch
import pandas as pd
import numpy as np
from datetime import datetime
import scipy.stats as stats
from tqdm import tqdm
import logging

logging.getLogger().setLevel(logging.INFO)


def read_variantsF(variants_path_file, device):
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
    variants_tensor = torch.as_tensor(variants, dtype=torch.float32, device=device)
    return subIDs, varIDs, variants_tensor, df


def read_variantsF_efficient(variants_path_file, file_name, device):
    """
    read the large genomic file (row_SNPs x column_subjects) using pandas
    if the genomic file is too large,can cover the csv file to pickle file.
    :param variants_path_file:
    :param variants_size: if variants_size=None,select all the variants
    :return: subIDs, varIDs, variants_tensor, df  # list, list, array and dataframe
    """
    varIDs_file = open(os.path.join(variants_path_file, "{}_varIDs.txt".format(file_name)), "r")
    varIDs = varIDs_file.read().split("\n")
    subIDs_file = open(os.path.join(variants_path_file, "{}_subIDs.txt".format(file_name)), "r")
    subIDs = subIDs_file.read().split("\n")
    subIDs = list(int(x) for x in subIDs)
    variants = np.load(os.path.join(variants_path_file, "{}_variants_value.npy".format(file_name)))

    A0 = np.ones(len(subIDs), dtype=np.int8)
    variants = np.row_stack((A0, variants))
    varIDs.insert(0, 'A0')
    # when variants data is large, df will consume a large amount of memory,
    # but df is no being used, set df=pd.Dataframe()
    df = pd.DataFrame()
    # df = pd.DataFrame(variants, index=varIDs, columns=subIDs, dtype=np.int8)
    variants_tensor = torch.tensor(variants, dtype=torch.float32, device=device)

    return subIDs, varIDs, variants_tensor, df


def read_traitsF(traits_path_file, device):
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
    traits_tensor = torch.as_tensor(traits.values, dtype=torch.float32, device=device)
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

def get_gpu_meomry_usage(device):
    """
    Print current memory allocation and peak memory allocation in GB
    :param device:
    :return:
    """
    logging.info("Current GPU Memory allocated: {:.2f} GB".format(torch.cuda.memory_allocated(device) / 1024 ** 3))
    logging.info("Peak GPU Memory allocated: {:.2f} GB".format(torch.cuda.max_memory_allocated(device) / 1024 ** 3))

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
    with open(os.path.join("..", "results", "Ch12wgs_multiTraits_sGD_020522_pytorch_without_data_slicing.csv"),
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


def cal_lgM(variants, variants_batch, weights_batch, traits, traits_xor, variant_type):
    """
    Calculate and return the lgM for all the drivers or any driver for any given population for multiple traits
    :param variants:
    :param variants_batch:
    :param weights_batch:
    :param traits:
    :param traits_xor:
    :param variant_type:
    :return:
    """
    # BP_V is DW, BP_V_xor is (J-D)W', traits_xor is J-traits, J is all one matrix
    # BP_V represents the weighted traits of all variants, weights_batch is b*n, traits is n*k,BP_V is b*n*k,
    # b is the batch_size
    BP_V = torch.einsum('ij,jk->ijk', weights_batch, traits)
    # BP_V_xor represents the weighted traits_xor of all variants, weights_batch is b*n, traits_xor is n*k,
    # BP_V_xor is b*n*k, b is the batch_size
    BP_V_xor = torch.einsum('ij,jk->ijk', weights_batch, traits_xor)
    # calculate the second dimension summation of BP_V, and add a broadcasting dimension, BP_V_sum is b*1*k
    BP_V_sum = BP_V.sum(dim=1).unsqueeze(dim=1)
    # calculate the second dimension summation of BP_V_xor, and add a broadcasting dimension, BP_V_xor_sum is b*1*k
    BP_V_xor_sum = BP_V_xor.sum(dim=1).unsqueeze(dim=1)
    # variant=1
    if variant_type == 1:
        # Unsqueeze variants_batch to add a broadcasting dimension: from (b, n) to (b, 1, n)
        variants_unsqueezed = variants_batch.unsqueeze(dim=1)
        # three-dimensional matrix multiplication, V1D1, V1D0, V0D1, V0D0 is b*1*k, for all the snps
        # V1D1 = torch.bmm(variants_unsqueezed, BP_V)
        # V1D0 = torch.bmm(variants_unsqueezed, BP_V_xor)
        V1D1 = torch.einsum("bqn,bnk->bqk",variants_unsqueezed, BP_V)
        V1D0 = torch.einsum("bqn,bnk->bqk",variants_unsqueezed, BP_V_xor)
        V0D1 = BP_V_sum - V1D1
        V0D0 = BP_V_xor_sum - V1D0
    # variant=0
    else:
        V1D1 = torch.einsum('mn,bnk->bmk', variants, BP_V)
        V1D0 = torch.einsum('mn,bnk->bmk', variants, BP_V_xor)
        V0D1 = torch.ones_like(V1D1) * BP_V_sum - V1D1
        V0D0 = torch.ones_like(V1D0) * BP_V_xor_sum - V1D0

    # if variant_type=1, lgM is b*1*k; if variant_type=0, lgM is b*m*k
    lgM = torch.lgamma(torch.tensor(2.0)) - torch.lgamma(2.0 + V0D1 + V0D0)
    lgM += torch.lgamma(1.0 + V0D0) - torch.lgamma(torch.tensor(1.0))
    lgM += torch.lgamma(1.0 + V0D1) - torch.lgamma(torch.tensor(1.0))

    # when j=1 (V=1)
    lgM += torch.lgamma(torch.tensor(2.0)) - torch.lgamma(2.0 + V1D1 + V1D0)
    lgM += torch.lgamma(1.0 + V1D0) - torch.lgamma(torch.tensor(1.0))
    lgM += torch.lgamma(1.0 + V1D1) - torch.lgamma(torch.tensor(1.0))


    # if variants_tensor.ndim == 1:
    #     lgM = lgM.reshape(1, lgM.shape[0])

    return lgM


def lgMcal(variants, traits, devices):
    """
    sGDsearch core code
    :param variants: m*n, m:the number of variants, n:the number of individuals
    :param traits:  n*k, n:the number of individuals, k:the number of traits
    :param devices:
    :return:
    """
    # set batch size
    batch_size = 256
    variants_num = variants.shape[0]
    split_size = int(np.ceil(variants_num / batch_size))
    logging.info("batch_size:{},split_size:{}".format(batch_size, split_size))
    offset = 0
    # element_run save sGD results
    element_run = []
    # cycle through each batch
    for i in tqdm(range(split_size)):
        # get variants and varIDs for each batch, variants_batch is b*n, b is the batch_size
        variants_batch = variants[offset:offset + batch_size]
        varIDs_batch = np.array(varIDs[offset:offset + batch_size]).reshape(-1, 1)
        # traits_xor is the inverse of traits, the 0 in traits become 1 in traits_xor, and
        # 1 in traits become 0 in traits_xor
        traits_xor = torch.ones(traits.shape, dtype=torch.float32, device=devices) - traits
        # variant=1
        weights_batch_v1 = variants_batch
        lgMv1_SD = cal_lgM(variants, variants_batch, weights_batch_v1, traits, traits_xor,variant_type=1)
        # remove one dimension,from (b,1,k) to (b,k)
        lgMv1_SD = lgMv1_SD.squeeze(dim=1)
        # released unused variable
        del weights_batch_v1
        torch.cuda.empty_cache()
        # variants=0
        # select individuals with variants=0 for all the snps
        weights_batch_v0 = torch.ones(variants_batch.shape, dtype=torch.float32, device=devices) - variants_batch
        lgMv0 = cal_lgM(variants, variants_batch, weights_batch_v0, traits, traits_xor,variant_type=0)
        # released unused variable
        del weights_batch_v0
        torch.cuda.empty_cache()
        # collect the lgMv0_topGD for each trait; the lgM value for V0 group when using topGD as the driver
        lgMv0_topGD = []
        # collect the r between SD and topGD for each trait
        r = []
        variants_batch_array = variants_batch.to("cpu").numpy()
        if use_oneTopGD:
            for m in range(0, len(traitIDs)):
                lgMv0_topGD.append(list(lgMv0[:, m, m].cpu().numpy()))
            for j in topGD_index:
                r_each_trait = []
                for row in variants_batch_array:
                    # [0] to get only the coefficient and ignore the p-values
                    r1 = stats.spearmanr(row, variants[j, :].to("cpu").numpy())[0]
                    r_each_trait.append(r1)  # a vector of K
                r.append(r_each_trait)
            lgMv0_sGD = torch.zeros((variants_batch.shape[0], len(traitIDs)), device=device)
            sGD = np.zeros((variants_batch.shape[0], len(traitIDs)))
        else:
            # with sGD, lgMv0 is bath_size*m_variants*k_traits
            lgMv0_sGD = torch.max(lgMv0, dim=1).values
            sGD_index = torch.max(lgMv0, dim=1).indices

            # collect the variant ID of sGD for each batch and each trait in a 1D array
            sGD = np.array([varIDs[i] for pair in sGD_index for i in pair]).reshape(sGD_index.shape)

            k = 0
            # collect the lgMv0_topGD and r for each trait in a 1D array specifically with mxk lgMv0
            # topGD_index is one output from GDsearch_all, a vector of K (#traits ordered in the original trait input file)
            for j in topGD_index:
                # a vector of K
                lgMv0_topGD.append(list(lgMv0[:, j, k].cpu().numpy()))

                r_each_trait = []
                for row in variants_batch_array:
                    # [0] to get only the coefficient and ignore the p-values
                    r1 = stats.spearmanr(row, variants[j, :].to("cpu").numpy())[0]
                    r_each_trait.append(r1)  # a vector of K
                r.append(r_each_trait)
                k = k + 1
        lgMv0_topGD = torch.tensor(np.array(lgMv0_topGD).T, device=device)
        r = np.array(r).T

        if use_oneTopGD:
            lgM_v1v0 = lgMv1_SD + lgMv0_topGD
        else:
            lgM_v1v0 = lgMv1_SD + lgMv0_sGD
        # get sequence number for varIDs_batch
        seq = np.arange(offset, offset + variants_batch.shape[0]).reshape(-1, 1)

        # save all the batch results to merged_arr
        merged_arr = np.concatenate([lgMv1_SD.cpu().numpy(), lgMv0_sGD.cpu().numpy()], axis=1)
        merged_arr = np.concatenate([merged_arr, lgMv0_topGD.cpu().numpy()], axis=1)
        merged_arr = np.concatenate([merged_arr, lgM_v1v0.cpu().numpy()], axis=1)
        merged_arr = np.concatenate([merged_arr, sGD], axis=1)
        merged_arr = np.concatenate([merged_arr, r], axis=1)
        merged_arr = np.concatenate([merged_arr, seq], axis=1)
        merged_arr = np.concatenate([merged_arr, varIDs_batch], axis=1)

        element_run.extend(merged_arr)
        offset += batch_size
    return element_run


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
    df_res = pd.DataFrame(element_run, columns=outAll)
    output_file_path = os.path.join("..", "results", "sGD_pytorch_remove_data_slicing_and_loop.csv")
    df_res.to_csv(output_file_path, index=False)


if __name__ == '__main__':
    # Create a device with a GPU. If you don't have a GPU, the code will produce an error.
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cpu')
    # release GPU memory that is no longer in use, this function clears the GPU memory cache,
    # making more memory available for other operations.

    # 1 read data
    start_time = datetime.now()
    # root_path = os.path.join("..", "data", "1M")
    root_path = os.path.join("..", "data", "90K")
    # root_path = os.path.join("..", "data", "test_data")
    # subIDs, varIDs, variants_tensor, df_variants = read_variantsF(
    subIDs, varIDs, variants_tensor, df_variants = read_variantsF_efficient(
        os.path.join(root_path), "90k", device)
    # os.path.join(root_path),"1M",device)
    # os.path.join(root_path, 'variants_test.csv'), device)

    subIDs_BP, traitIDs, traits_tensor = read_traitsF(
        # os.path.join(root_path, 'Phenotype_KidsFirst_Index01.csv'),device)
        os.path.join(root_path, 'Phenotype_exonic_01.csv'), device)
    # os.path.join(root_path, 'traits_test.csv'), device)

    end_time = datetime.now()
    elapsed_time = (end_time - start_time).seconds
    logging.info("-" * 30 + "start read data" + "-" * 30)
    logging.info(str(end_time))
    logging.info('read data elapsed time: {}s'.format(elapsed_time))
    logging.info("variants:{}".format(np.shape(variants_tensor)))
    logging.info("traits: {}".format(np.shape(traits_tensor)))
    get_gpu_meomry_usage(device)

    # 2 With GDsearch_all, calculate and output the global stats related to all the traits for all the variants

    logging.info("-" * 30 + "start GDsearch all" + "-" * 30)
    torch.cuda.empty_cache()
    get_gpu_meomry_usage(device)

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
    # An important flag to dictate whether using topGD or sGD as the driver for A0 group.
    logging.info("-" * 30 + "start sGD search" + "-" * 30)
    del rr, glgm, topGD, glgm_topGD
    torch.cuda.empty_cache()
    get_gpu_meomry_usage(device)
    use_oneTopGD = False
    start_time = time.time()
    logging.info("device:{}".format(device))

    element_run = lgMcal(variants_tensor, traits_tensor, device)
    logging.info("sGD elapsed time: {}s".format(time.time() - start_time))
    save_sGD_result(element_run)
