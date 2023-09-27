# -*- coding: utf-8 -*-

# @File    : IBI.py
# @Date    : 2023-08-22
# @Author  : ${ RenJin}

### ICI implementation in python (Global Driver Search using matrix operations): Jinling Liu 12/26/2021
### Can be applied to multiple traits or single trait; ready to be run on server after converted to .py file.
### 02-09-22 updated the function of lgM_cal_1 so it can either calculate using topGD or sGD, with a flag of "use_topGD" before that function
### 05-01-22 corrected in the function of lgM_cal_1 the final line of calculating lgMv1v0 thus it was calculated differently using topGD or sGD
# and rename this as lgM_cal and deleted the commented codes.


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# from matplotlib_venn import venn2,venn2_circles
import json
import os
import shutil
import scipy.stats as stats
from scipy.stats import fisher_exact
import scipy
from datetime import datetime
import math
import time
from joblib import Parallel, delayed


def read_traitsF(traitsF):  ### read the .csv file with traits (subjects_row x traits_column)
    traits = pd.read_csv(traitsF,
                         index_col=0)  # make sure the subIDs become the index column; as default, column names are inferred from the first line of the file
    subIDs = list(traits.index)
    traitIDs = traits.columns
    traits = np.array(traits,
                      dtype=np.int16)  # np.int8 has changed the type to int8; when using int8, the subIDs become negative.
    #     print(np.sum(traits)) # sum gives odd result of -94.
    return (subIDs, traitIDs, traits)  # list, array


def read_variantsF(variantsF):  ## read the large genomic file (row_SNPs x column_subjects) line by line
    header = True
    varaiants = []
    variants = []
    with open(variantsF) as f:
        for line in f:  # go through each SNP
            if header:  # extract the head of subject IDs
                subIDs = line.strip().split(',')
                subIDs = list(subIDs[1:])
                header = False
                continue
            line = line.strip().split(
                ',')  # vID = line[0] ,the variant ID; line1 = line[1:] ,the SNP state values for all the subjects
            variants.append(line)
    variants = np.array(variants)
    varIDs = list(variants[:, 0])
    variants = variants[:, 1:].astype(int)  # int or np.int8 does not work and variants stay in the type of float 64.??
    A0 = np.ones(len(subIDs))
    variants = np.row_stack((A0, variants))
    varIDs.insert(0, 'A0')
    return (subIDs, varIDs, variants)  # list, list and array


def read_variantsF1(variantsF):  ## read the large genomic file (row_SNPs x column_subjects) using pandas
    df = pd.read_csv(variantsF,
                     index_col=0)  # this will turn the first column of varIDs into index thus df.columns will only include subIDs
    # TODO select small dataset
    df=df.iloc[:2,:]
    varIDs = list(df.index)
    subIDs = list(int(x) for x in df.columns)
    variants = np.array(df, dtype=np.int8)  # Somehow, np.int8 does not work here.
    A0 = np.ones(len(subIDs), dtype=np.int8)
    variants = np.row_stack((A0, variants))
    varIDs.insert(0, 'A0')
    df = pd.DataFrame(variants, index=varIDs, columns=subIDs, dtype=np.int8)
    return (subIDs, varIDs, variants, df)  # list, list, array and dataframe


def DriverSearch(traits, variants):
    ###Calcuate and return the lgM for all the drivers or any driver for any given population for multiple traits
    ###Get the max/min GD and SD as well as their lgM; this can be done in the lgM_cal function so this function can stay the same
    ###Get the nxk matrix of traits==0 and the nxk matrix of traits==1 (n,#subjects;k,#traits; thus capable of working with multipe traits)
    ###if no individuals are in V0 group when the passed variants is [], the V0D0 counts as well as lgM will be 0; the max value/index are both turned as 0
    ###no other SNPs except A0 have a constant value since those have been removed in the preprocessing step;
    #     print(variants.shape) #(1442,) if only variant is passed in.
    #     print(variants.ndim)

    bpMask0 = traits == 0
    bpMask0 = bpMask0.astype(np.int16)
    d0 = np.sum(bpMask0)  # 930 HTN and 4360 non-HTN making a totla of 5290 subjects

    bpMask1 = traits == 1
    bpMask1 = bpMask1.astype(np.int16)
    d1 = np.sum(bpMask1)

    ### Get the mxn vector of snp==0 and the mxn vector of snp==1
    snpMask0 = variants == 0
    snpMask0 = snpMask0.astype(np.int16)  # print(snpMask0.sum(axis=1)) print(snpMask1.shape)

    snpMask1 = variants == 1
    snpMask1 = snpMask1.astype(np.int16)

    ### Get the four mx1 vector as below: m is # of SNPs in the dataset; for each SNP, the corresponding 4 values
    # from the 4 vectors make up the 2x2 tables between SNP and hypertension
    # lgM variants_num × traits_num
    V0D0 = snpMask0 @ bpMask0  # snpMask0, variants_row x subjects_column
    V1D0 = snpMask1 @ bpMask0  # bpMask0, subjects_row x traits_column
    V0D1 = snpMask0 @ bpMask1
    V1D1 = snpMask1 @ bpMask1

    ### Calculate the log Marginal LikelihooD for all the SNPs in the matrix based on the collected counts and equation 5 in the worD file
    # when j=0 (V=0)
    lgM = scipy.special.loggamma(2.0) - scipy.special.loggamma(2.0 + V0D1 + V0D0)
    lgM += scipy.special.loggamma(1.0 + V0D0) - scipy.special.loggamma(1.0)
    lgM += scipy.special.loggamma(1.0 + V0D1) - scipy.special.loggamma(1.0)

    # when j=1 (V=1)
    lgM += scipy.special.loggamma(2.0) - scipy.special.loggamma(2.0 + V1D1 + V1D0)
    lgM += scipy.special.loggamma(1.0 + V1D0) - scipy.special.loggamma(1.0)
    lgM += scipy.special.loggamma(1.0 + V1D1) - scipy.special.loggamma(1.0)

    if variants.ndim == 1:
        lgM = lgM.reshape(1, lgM.shape[0])  # lgM is #traits x 1;

    return (lgM)  # lgM is a 2D array of #variants x #traits with print(np.shape(lgM))


def GDsearch_all(traits, variants):
    ## Get all the stats for all the variants in any given population for multiple traits; particulary used for the entire population
    ## Get the nxk matrix of traits==0 and the nxk matrix of traits==1 (n,#subjects;k,#traits)
    ## if no individuals are in V0 group when the passed variants is [], the V0D0 counts as well as lgM will be 0.

    bpMask0 = traits == 0
    bpMask0 = bpMask0.astype(np.int16)
    d0 = np.sum(bpMask0)

    bpMask1 = traits == 1
    bpMask1 = bpMask1.astype(np.int16)
    d1 = np.sum(bpMask1)

    ### Get the mxn vector of snp==0 and the mxn vector of snp==1
    snpMask0 = variants == 0
    snpMask0 = snpMask0.astype(np.int16)

    snpMask1 = variants == 1
    snpMask1 = snpMask1.astype(np.int16)

    ### Get the four mx1 vector as below: m is # of SNPs in the dataset; for each SNP, the corresponding 4 values
    # from the 4 vectors make up the 2x2 tables between SNP and hypertension
    # Jin V1D1:p(V=1,D=1)calculate the Snp(V)=1 and tarit(D)=1，each Snp's individual num
    V0D0 = snpMask0 @ bpMask0
    V1D0 = snpMask1 @ bpMask0
    V0D1 = snpMask0 @ bpMask1
    V1D1 = snpMask1 @ bpMask1

    ### GiVen the Dirichlet Distributions we are using, the expectation of these conditional probabilities is as follows: prior probability
    # Jin cp_D1V1=p(D=1|V=1),cp_D1V0=p(D=1|V=0)
    cp_D1V1 = (1 + V1D1) / (
            2 + V1D1 + V1D0) * 1.0  # P(D=1|V=1) = (alpha11 + V1D1)/(alpha1 + V1D1 + V1D0)*1.0
    cp_D1V0 = (1 + V0D1) / (
            2 + V0D1 + V0D0) * 1.0  # P(D=1|V=0) = (alpha01 + V0D1)/(alpha0 + V0D1 + V0D0)*1.0
    RR = cp_D1V1 / cp_D1V0  # RR is risk ratio; OR is oDDs ratio

    ### Calculate the log Marginal LikelihooD for this particular SNP based on the collected counts and equation 5 in the worD file
    # when j=0 (V=0)
    lgM = scipy.special.loggamma(2.0) - scipy.special.loggamma(2.0 + V0D1 + V0D0)
    lgM += scipy.special.loggamma(1.0 + V0D0) - scipy.special.loggamma(1.0)
    lgM += scipy.special.loggamma(1.0 + V0D1) - scipy.special.loggamma(1.0)

    # when j=1 (V=1)
    lgM += scipy.special.loggamma(2.0) - scipy.special.loggamma(2.0 + V1D1 + V1D0)
    lgM += scipy.special.loggamma(1.0 + V1D0) - scipy.special.loggamma(1.0)
    lgM += scipy.special.loggamma(1.0 + V1D1) - scipy.special.loggamma(1.0)

    if variants.ndim == 1:
        lgM = lgM.reshape(1, lgM.shape[0])  # lgM is #traits x 1;otherwise, lgM is, variants x traits.
    max_value = np.max(lgM,
                       axis=0)  # get the max and index of TopGD across all the rows of variants for each column of the trait inside the 2-D array
    max_index = np.argmax(lgM, axis=0)  # thus, max_value or max_index is, one vector with the size of K (# of traits)

    return (RR, lgM, max_value, max_index)

# function Ms1+Mr0
def lgMcal(varID):  ## use DriverSearch for lgMv0 and for lgMv1 ## designed for using both one topGD and sGD
    i = varIDs.index(
        varID)  # when varIDs is used in the final paralleing code; #print(i) # not sure why this would not print if using the above parallel code
    index1 = variants[i, :] == 1  # identify the index of patients that have this particular variant Vs=1
    index0 = variants[i, :] == 0

    V1 = variants[:,
         index1]  # V1 = variants[i,index1] # 1 SNP X 2478 subjects who have v=1 for this SNP i; #V1wTopGD = variants[9980,index1] #array([9980]),'ss74821773_G', is the topGD for HTN in the 5290 x 38K file
    if use_oneTopGD:
        V0 = variants[topGD_index][:,
             index0]  # we will only consider and search over all the unique topGDs from all the traits;
        # thus one topGD for trait1 may be selected as the sGD for trait2;variants[topGD_index] is mxn
    else:
        V0 = variants[:, index0]  # V0 will be [] and its shape will be (0,) if index0 is all false

    BP_V1 = traits[
        index1]  # 2478 subjects' hypertension status who have v=1 for this SNP （2478,) and may have HTN=0 or HTN=1
    BP_V0 = traits[index0]  # (5290-2478) subjects' hypertension status(1 or 0) who have v=0 for this SNP

    lgMv1_SD = DriverSearch(BP_V1, variants[i, index1])[0]
    # this should be as efficient as SD_lgM_V1; only calculates one marginal assuming SD as the cause, P(D|SD->HT)
    # with [0], the original 2D array, array([[-3127.91831177,...]]),
    # becomes the format of 1D array, array([-3127.91831177,...]),thus consistent with the other output values
    lgMv0 = DriverSearch(BP_V0, V0)  # lgM_v0 is the 2D array; kxk if topGD; m_variants x k_traits if sGD

    lgMv0_topGD = []  # collect the lgMv0_topGD for each trait in a 1D array; the lgM value for V0 group when using topGD as the driver
    r = []  # collect the r between SD and topGD for each trait in a 1D array
    if use_oneTopGD:  # collect the lgMv0_topGD and r for each trait in a 1D array specifically with kxk lgMv0
        for m in range(0, len(traitIDs)):
            lgMv0_topGD.append(lgMv0[m, m])  # with oneTOPGD, lgMv0 is kxk,since k top GD for k traits; here it selects
            # the values of P(D0|topGD-k -> trait-k);
        for j in topGD_index:  # topGD_index is a global variable obtained outside this function
            r1 = stats.spearmanr(variants[i, :], variants[j, :])[0]
            r.append(r1)
        lgMv0_sGD = np.zeros(len(traitIDs))
        sGD = np.zeros(len(traitIDs))
    else:
        lgMv0_sGD = np.max(lgMv0, axis=0)  # with sGD, lgMv0 is m_variants x k_traits
        sGD_index = np.argmax(lgMv0, axis=0)

        sGD = []  # collect the variant ID of sGD for each trait in a 1D array
        for item in sGD_index:
            sGD.append(varIDs[item])
        sGD = np.array(sGD)

        k = 0  # collect the lgMv0_topGD and r for each trait in a 1D array specifically with mxk lgMv0
        for j in topGD_index:  # topGD_index is one output from GDsearch_all, a vector of K (#traits ordered in the original trait input file)
            lgMv0_topGD.append(lgMv0[j, k])  # a vector of K
            r1 = stats.spearmanr(variants[i, :], variants[j, :])[
                0]  # [0] to get only the coefficient and ignore the p-values
            r.append(r1)  # a vector of K
            k = k + 1
    lgMv0_topGD = np.array(lgMv0_topGD)
    r = np.array(r)

    if use_oneTopGD:
        lgM_v1v0 = lgMv1_SD + lgMv0_topGD
    else:
        lgM_v1v0 = lgMv1_SD + lgMv0_sGD

    return (lgMv1_SD, lgMv0_sGD, lgMv0_topGD, lgM_v1v0, sGD, r, i, varID)


### Read the files to get matrix of variants and traits;
root_path = os.path.join("..","data")
clock_start = datetime.now()  # 46 seconds total to read the 38Kx5290 file and to run GDsearch
start = time.perf_counter()  # 39s to read the above file using time.perf_counter()
print(clock_start)

### 10-SNPs_strictFilteration_T.csv
# subIDs, varIDs, variants, df_variants = read_variantsF1 ('/Volumes/GoogleDrive/My Drive/jupyter_notebooks/NHLBI/pyICI_11242021v/inputs/10-SNPs_strictFilteration_T.csv')
### 38K-SNPs_strictFilteration_T.csv
# subIDs, varIDs, variants, df_variants = read_variantsF1 ('/Volumes/GoogleDrive/My Drive/jupyter_notebooks/NHLBI/pyICI_11242021v/inputs/38K-SNPs_strictFilteration_T.csv')
### BP_HTN
# subIDs_BP, traitIDs, traits = read_traitsF('/Volumes/GoogleDrive/My Drive/jupyter_notebooks/NHLBI/pyICI_11242021v/inputs/BP_ICI_newfiltration_50k_TrainingSet.csv')

### chr12 WGS FHS; F1 took 20 minutes to read in these 438K SNPs x 4111 subs on my mac and 245s on Foundry terminal.
subIDs, varIDs, variants, df_variants = read_variantsF1(
    os.path.join(root_path , 'chrm21__KidsFirst_snp01_dominant_withCorrect_Index_RR1.csv'))
### BP_HTN_WGS
subIDs_BP, traitIDs, traits = read_traitsF(os.path.join(root_path , 'Phenotype__KidsFirst_withCorrect_Index.csv'))

### 10SNPs_mRNA
# subIDs, varIDs, variants, df_variants = read_variantsF1 ('/Volumes/GoogleDrive/My Drive/jupyter_notebooks/NHLBI/pyICI_11242021v/inputs/10SNPs_mRNA_BP.csv')
### 38K_SNPs mRNA
# subIDs, varIDs, variants, df_variants = read_variantsF1 ('/Volumes/GoogleDrive/My Drive/jupyter_notebooks/NHLBI/pyICI_11242021v/inputs/SNPs_DominantCoding_newfiltration_fullset_38k_for_mRNAandBP.csv')
### BP_mRNA
# subIDs_BP, traitIDs, traits = read_traitsF("/Volumes/GoogleDrive/My Drive/jupyter_notebooks/NHLBI/pyICI_11242021v/inputs/mRNAtop9DEGs_byDiscretizeKmeans_and_BpHTN_traits_v2.csv")

print(np.shape(variants))  # print(variants.dtype)
print(np.shape(traits))
print(
    subIDs == subIDs_BP)  ##This is to double check the dimensions are correct, and you would want to look for "True" for this.

varIDs_str = [] ## use this when the varIDs are not str but numbers which should be avoided.
for item in varIDs:
     varIDs_str.append(str(item))

clock_stop = datetime.now()
elapsed_time = time.perf_counter() - start
print(str(clock_stop) + '; total time used, ' + str(elapsed_time))

### With GDsearch_all, calculate and output the global stats related to all the traits for all the variants using the entire population

clock_start = datetime.now()  # 46 seconds total to read the 38Kx5290 file and to run GDsearch
start = time.perf_counter()  # 39s to read the above file using time.perf_counter()
print(clock_start)

rr, glgm, glgm_topGD, topGD_index = GDsearch_all(traits, variants)  # rr is relative risk
# the variants order in the original variants file (thus in this 2D array of variants) is kept the same
# among all the outputs from GDsearch_all,DriverSearch and varIDs..

## collect the headers for the output file
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

## output the RR and glgm for all the variants
with open("Ch12wgs_multiTraits_GDsearch_020922.csv",
          "w") as outfile:  # more efficient than using dataframe to_csv...
    outfile.write(','.join(gstat_newhead) + '\n')
    for i in range(0, rr.shape[0]):
        ls = []
        ls.extend(rr[i].tolist())  # row i of rr that is corresponding to the ith variant
        ls.extend(glgm[i].tolist())
        ls.extend([str(i), varIDs[i]])
        outfile.write(','.join(str(item) for item in ls) + '\n')

clock_stop = datetime.now()
elapsed_time = time.perf_counter() - start
print(str(clock_stop) + '; total time used, ' + str(elapsed_time))

### Get and output the topGD for each trait using GDsearch_all

topGD = []
for item in topGD_index:
    topGD.append(
        varIDs[item])  # currently the wgs SNPs are labeled with numbers, thus varIDs and topGD both are int lists.

# uniGD_index = list(set(topGD_index))

with open("Ch12wgs_multiTraits_GDsearch-topGD_020922.csv", "w") as outfile:
    for i in range(0, len(traitIDs)):
        line = [traitIDs[i], str(topGD[i]), str(glgm_topGD[i])]
        #         print(line)
        outfile.write(','.join(str(item) for item in line) + '\n')

### paralleling run all the 38K SNPs with the function of cal_lgM(i) and write these results into a .csv file;
### the total time (8.5 hs) is obtained for sGD 19276 SNPS with paraleling on this computer (8 core/32G); the orginal time was 51hs.

use_oneTopGD = False  # An important flag to dictate whether using topGD or sGD as the driver for A0 group.

clock_start = datetime.now()  # 46 seconds total to read the 38Kx5290 file and to run GDsearch
start = time.perf_counter()  # 39s to read the above file using time.perf_counter()
print(clock_start)

# TODO
# for var in varIDs[0:100]:
#     lgMcal(var)

# For n_jobs below -1, (n_cpus + 1 + n_jobs) are used
element_run = Parallel(n_jobs=-1)(delayed(lgMcal)(var) for var in varIDs)  # bigRR_varID[0:50]
# print(len(element_run))

clock_stop1 = datetime.now()
elapsed_time = time.perf_counter() - start
print(str(clock_stop1) + '; total time used, ' + str(elapsed_time))

### 118 seconds used for 100 SNPs by using topGD for the dataset of 3 traits x 428K SNPs, thus 140 hs on this mac
### 9 hs were used for topGD WGS Ch12(3 traits x 428K SNPs)on Foundry (~32 nodes)

## collect the headers for this file
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
with open("Ch12wgs_multiTraits_sGD_020522.csv", "w") as outfile:
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

print(datetime.now())

