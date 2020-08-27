#coding:utf-8

import csv
import os
import shutil
import sys
import numpy as np
from sklearn.model_selection import KFold

g_ad_file_name = "/Users/check4068/Desktop/腾讯算法大赛/train_preliminary/ad.csv"
g_click_file_name = "/Users/check4068/Desktop/腾讯算法大赛/train_preliminary/click_log.csv"
g_invalid_creative_set = set()
g_ad_header = ''
g_click_header = ''



def mkdir(path):
    path = path.strip()
    path = path.rstrip("/")

    if os.path.exists(path):
        print(path + ' already exist, now delete')
        shutil.rmtree(path)

    os.makedirs(path)
    print(path + ' created success')


def mk_csv_file(source_file_name, path, data):
    global g_ad_header
    global g_click_header


    csv_file = open(path, 'wb')
    try:
        writer = csv.writer(csv_file)
        # if source_file_name == "ad.csv":
        #     writer.writerow(g_ad_header)
        # if source_file_name == "click_log.csv":
        #     writer.writerow(g_click_header)

        print(len(data))
        for _line in data:
            writer.writerow(_line)
    finally:
        print(path + "csv created success, ")
        csv_file.close()


def read_file(file_name):
    global g_ad_header
    global g_click_header
    global g_ad_file_name
    global g_click_file_name
    global g_invalid_creative_set

    with open(file_name, "r") as f:
        file_data = f.readlines()

    data_list = []
    if file_name == g_ad_file_name:
        g_ad_header = file_data[0].replace("\n", "").split(',')
        count = 0
        for _line in file_data[1:]:
            count += 1
            if count > 100:
                break
            creative_id, ad_id, product_id, product_category, advertiser_id, industry = _line.replace("\n", "").split(',')
            if product_id == "/N" or industry == "/N":
                g_invalid_creative_set.add(creative_id)
                continue
            data_list.append([creative_id, ad_id, product_id, product_category, advertiser_id, industry])

    if file_name == g_click_file_name:
        g_click_header = file_data[0].replace("\n", "").split(',')
        count = 0
        for _line in file_data[1:]:
            count += 1
            if count > 100:
                break
            time, user_id, creative_id, click_times = _line.replace("\n", "").split(',')
            if int(click_times) > 1:
                continue
            if creative_id in g_invalid_creative_set:
                continue
            data_list.append([time, user_id, creative_id, click_times])

    return data_list


def pre_treament(num):
    global g_ad_file_name
    global g_click_file_name

    k_fold_valid(g_ad_file_name, num)
    k_fold_valid(g_click_file_name, num)


def k_fold_valid(file_name, num):
    data_list = read_file(file_name)
    X = np.array(data_list)

    f_dir = os.getcwd() + "/" + file_name.replace('.', '_') + "_divide"
    mkdir(f_dir)

    count = 1
    kf = KFold(n_splits=num)
    for train_index, test_index in kf.split(X):
        s_dir = f_dir + "\\divider_" + str(count)
        mkdir(s_dir)

        mk_csv_file(file_name, s_dir + "\\train_data.csv", X[train_index])
        mk_csv_file(file_name, s_dir + "\\test_data.csv", X[test_index])

        count += 1



if __name__ == "__main__":
    divide_num = 5
    if len(sys.argv) >= 2:
      divide_num = int(sys.argv[1])

    pre_treament(divide_num)