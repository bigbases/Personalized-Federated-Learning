# PFLlib: Personalized Federated Learning Algorithm Library
# Copyright (C) 2021  Jianqing Zhang

# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.

import h5py
import numpy as np
import os


def average_data(algorithm="", dataset="", data_dir="", goal="", model_name="", prev = 0, times=10):
    test_acc = get_all_results_for_one_algo(algorithm, dataset, data_dir, goal, model_name, prev, times)

    #여러번 실행 시 통계량
    max_accurancy = []
    for i in range(times - prev):
        max_accurancy.append(test_acc[i].max())

    print("std for best accurancy:", np.std(max_accurancy))
    print("mean for best accurancy:", np.mean(max_accurancy))


def get_all_results_for_one_algo(algorithm="", dataset="", data_dir="", goal="", model_name="", prev = 0, times=10):
    test_acc = []
    algorithms_list = [algorithm] * times
    for i in range(prev, times):
        if algorithms_list[i].split('_')[0] == "FedTest":
            file_name1 = algorithms_list[i] + "_" + model_name
            file_name2 = goal + "_" + str(i)
            test_acc.append(np.array(read_data_then_delete(file_name1, file_name2, data_dir, my = True, delete=False)))
        else:
            file_name1 = algorithms_list[i] + "_" + model_name
            file_name2 = goal + "_" + str(i)
            test_acc.append(np.array(read_data_then_delete(file_name1, file_name2, data_dir, my = False, delete=False)))

    # for i in range(times):
    #     file_name1 = "_" + algorithms_list[i] + "_" + model_name
    #     file_name2 = goal + "_" + str(i)
    #     test_acc.append(np.array(read_data_then_delete(file_name1, file_name2, data_dir, my = True, delete=False)))

    return test_acc


def read_data_then_delete(file_name1, file_name2, data_dir, my, delete=False):
    if my:
        file_path = "./results/" + data_dir.split('/')[-1] + "/" + file_name1 + "/test/" + file_name2 + ".h5"
    else:
        file_path = "./results/" + data_dir.split('/')[-1] + "/" + file_name1 + "/" + file_name2 + ".h5"
        
    with h5py.File(file_path, 'r') as hf:
        rs_test_acc = np.array(hf.get('rs_test_acc'))

    if delete:
        os.remove(file_path)

    return rs_test_acc