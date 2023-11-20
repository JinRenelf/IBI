# -*- coding: utf-8 -*-

# @File    : FileIO.py
# @Date    : 2023-10-05
# @Author  : ${ RenJin}

import os
import time
import numpy as np
import pandas as pd
from tqdm import tqdm


def make_dirs(folder_path):
    """
    make directory
    :param folder_path:
    :return:
    """
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"The folder was created successfully at {folder_path}")
    else:
        print("The folder already exists.")


def compress_file(file_path, output_file_folder, file_name):
    st=time.time()
    df = pd.read_csv(file_path, index_col=0)
    print("read orig data elpased time:{}".format(time.time()-st))
    subIDs = list(df.columns)
    data = np.array(df, dtype=np.int8)
    varIDs = df.index
    with open(os.path.join(output_file_folder, '{}_subIDs.txt'.format(file_name)), 'w') as file:
        file.write('\n'.join(subIDs))

    with open(os.path.join(output_file_folder, '{}_varIDs.txt'.format(file_name)), 'w') as file:
        file.write('\n'.join(varIDs))

    np.save(os.path.join(output_file_folder, "{}_variants_value.npy".format(file_name)), data)


def load_data(file_path, file_name):
    """
    load npy data
    :param file_path:
    :param file_name:
    :return:
    """
    variants = np.load(file_path + "{}_variants_value.npy".format(file_name))
    varIDs_file = open(file_path + '{}_varIDs.txt'.format(file_name), "r")
    varIDs = varIDs_file.read()
    varIDs = varIDs.split('\n')
    subIDs_file = open(os.path.join(file_path,'{}_subIDs.txt'.format(file_name)), "r")
    subIDs = subIDs_file.read().split("\n")
    return variants,varIDs,subIDs


if __name__ == '__main__':
    # test 1
    file_path=os.path.join("..","data","90k","exonic_variants_01.csv")
    compress_file(file_path,os.path.join("..","data","90k"), "90k")

    # folder_root_path = os.path.join("..","data", "all_chromsomes", "CHD_All_Chrm_variant_Files")
    # output_file_folder = os.path.join("..","data", "all_chromsomes", "CHD_All_Chrm_variant_Files_compressed")
    # make_dirs(output_file_folder)
    #
    # file_list = os.listdir(folder_root_path)
    # for file_name in tqdm(file_list):
    #     compress_file(os.path.join(folder_root_path,file_name), output_file_folder, file_name.split(".")[0])
