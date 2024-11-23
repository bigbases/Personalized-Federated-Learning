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

import os
import ujson
import numpy as np
import gc
from sklearn.model_selection import train_test_split
import random

def random_split(total, num_parts):
  proportions = np.random.rand(num_parts)
  proportions /= proportions.sum()

  split_values = np.floor(total * proportions).astype(int)

  if split_values.sum() < total:
    split_values[-1] += total - split_values.sum()

  return split_values.tolist()

def dir_split(least_samples, num_clients, dataset_label, alpha, dataidx_map, K, N, min_size, num_classes):
    try_cnt = 1
    while min_size < least_samples:
        if try_cnt > 1:
            print(f'Client data size does not meet the minimum requirement {least_samples}. Try allocating again for the {try_cnt}-th time.')

        idx_batch = [[] for _ in range(num_clients)]
        extra_samples = [[] for _ in range(num_classes)]
        for k in range(K):
            idx_k = np.where(dataset_label == k)[0]
            np.random.shuffle(idx_k)

            minimum = 30 * num_clients * num_classes
            rest = N % minimum / num_classes
 
            temp_cls_index = idx_k[:int(rest)]
            extra_samples[k] = list(temp_cls_index)
            print(N)
            print(minimum)
            print(f"(N - (N % minimum))/num_clients: {(N - (N % minimum))/num_clients}")

            idx_k = idx_k[int(rest):]
            proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
            proportions = np.array([p*(len(idx_j)<(N - (N % minimum))/num_clients) for p,idx_j in zip(proportions,idx_batch)])
            proportions = proportions/proportions.sum()
            proportions = (np.cumsum(proportions)*len(idx_k)).astype(int)[:-1]
            idx_batch = [idx_j + idx.tolist() for idx_j,idx in zip(idx_batch,np.split(idx_k,proportions))]
            min_size = min([len(idx_j) for idx_j in idx_batch])

        try_cnt += 1

    for j in range(num_clients):
        dataidx_map[j] = idx_batch[j]

    return dataidx_map, extra_samples

def check(config_path, train_path, test_path):
    # check existing dataset
    if os.path.exists(config_path):
        print("\nDataset already generated.\n")
        return True

    dir_path = os.path.dirname(train_path)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    dir_path = os.path.dirname(test_path)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    return False

def separate_data(data, num_clients, num_classes, alpha, least_samples, niid=False, balance=False, partition=None, class_per_client=None):
    X = [[] for _ in range(num_clients)]
    y = [[] for _ in range(num_clients)]
    statistic = [[] for _ in range(num_clients)]

    dataset_content, dataset_label = data

    dataidx_map = {}

    if not niid:
        partition = 'pat'
        class_per_client = num_classes

    if partition == 'pat':
        idxs = np.array(range(len(dataset_label)))
        idx_for_each_class = []
        for i in range(num_classes):
            idx_for_each_class.append(idxs[dataset_label == i])

        class_num_per_client = [class_per_client for _ in range(num_clients)]
        for i in range(num_classes):
            selected_clients = []
            for client in range(num_clients):
                if class_num_per_client[client] > 0:
                    selected_clients.append(client)

            selected_clients = selected_clients[:int(np.ceil((num_clients/num_classes)*class_per_client))]
            print(f"Class {i} is assigned to clients: {selected_clients}")
            
            if num_classes == 10:
                num_all_samples = len(idx_for_each_class[i]) - (len(idx_for_each_class[i]) % (60*num_classes))
                print(f"Class {i} has {num_all_samples} samples.")
            else:
                num_all_samples = len(idx_for_each_class[i])
                print(f"Class {i} has {num_all_samples} samples.")

            num_selected_clients = len(selected_clients)

            num_per = num_all_samples / num_selected_clients
            if balance:
                num_samples = [int(num_per) for _ in range(num_selected_clients-1)]
            else:
                num_samples = np.random.randint(max(num_per/num_classes, least_samples/num_classes), num_per, num_selected_clients-1).tolist()
            num_samples.append(num_all_samples-sum(num_samples))

            idx = 0
            for client, num_sample in zip(selected_clients, num_samples):
                if client not in dataidx_map.keys():
                    dataidx_map[client] = idx_for_each_class[i][idx:idx+num_sample]
                else:
                    dataidx_map[client] = np.append(dataidx_map[client], idx_for_each_class[i][idx:idx+num_sample], axis=0)
                idx += num_sample
                class_num_per_client[client] -= 1

    elif partition == "dir":
        # https://github.com/IBM/probabilistic-federated-neural-matching/blob/master/experiment.py
        min_size = 0
        K = num_classes
        N = len(dataset_label)

        dataidx_map, extra_samples = dir_split(least_samples, num_clients, dataset_label, alpha, dataidx_map, K, N, min_size, num_classes)

    else:
        print('In non-IID condition, Partition must be "pat" or "dir".')
        raise NotImplementedError

    if partition == "dir" and num_classes != 100:
        for client in range(num_clients):
            idxs = dataidx_map[client]
            X[client] = dataset_content[idxs]
            y[client] = dataset_label[idxs]
            unique_values = [item for item in set(y[client]) if list(y[client]).count(item) == 1]
            print(client, len(extra_samples))
            for value in unique_values:
                addindex = extra_samples[value][client]
                addX = dataset_content[addindex].reshape(1, 3, 32, 32)
                # addX = dataset_content[addindex].reshape(1, 1, 28, 28)
                addy = dataset_label[addindex].reshape(1)
                X[client] = np.append(X[client], addX, axis=0)
                y[client] = np.append(y[client], addy, axis=0)
            
            for i in np.unique(y[client]):
                statistic[client].append((int(i), int(sum(y[client]==i))))
                
        del data

        for client in range(num_clients):
            print(f"Client {client}\t Size of data: {len(X[client])}\t Labels: ", np.unique(y[client]))
            print(f"\t\t Samples of labels: ", [i for i in statistic[client]])
            print("-" * 50)

        return X, y, statistic

    else:
        for client in range(num_clients):
            idxs = dataidx_map[client]
            X[client] = dataset_content[idxs]
            y[client] = dataset_label[idxs]
            for i in np.unique(y[client]):
                statistic[client].append((int(i), int(sum(y[client]==i))))
                
        del data

        for client in range(num_clients):
            print(f"Client {client}\t Size of data: {len(X[client])}\t Labels: ", np.unique(y[client]))
            print(f"\t\t Samples of labels: ", [i for i in statistic[client]])
            print("-" * 50)
        return X, y, statistic

def split_data(X, y, train_size):
    # Split dataset
    train_data, test_data = [], []
    num_samples = {'train':[], 'test':[]}

    for i in range(len(y)):
        X_train, X_test, y_train, y_test = train_test_split(
            X[i], y[i], train_size=train_size, shuffle=True)

        train_data.append({'x': X_train, 'y': y_train})
        num_samples['train'].append(len(y_train))
        test_data.append({'x': X_test, 'y': y_test})
        num_samples['test'].append(len(y_test))

    print("Total number of samples:", sum(num_samples['train'] + num_samples['test']))
    print("The number of train samples:", num_samples['train'])
    print("The number of test samples:", num_samples['test'])
    print()

    del X, y
    # gc.collect()

    return train_data, test_data

def split_data_proportion(X, y, train_ratio):
    train_data, test_data = [], []
    num_samples = {'train':[], 'test':[]}

    for i in range(len(y)):
        xtrain, ytrain, xtest, ytest = None, None, None, None
        classes = np.unique(y[i])
        out = []
        for cls in classes:
            idx = np.where(y[i] == cls)[0]
            if len(idx) < 2:
                out.append(cls)
                continue
            X_cls = X[i][idx]
            y_cls = y[i][idx]
            X_train = X_cls[:int(len(X_cls)*train_ratio)]
            y_train = y_cls[:int(len(X_cls)*train_ratio)]
            X_test = X_cls[int(len(X_cls)*train_ratio):]
            y_test = y_cls[int(len(X_cls)*train_ratio):]

            if xtrain is None:
                xtrain = X_train
            else:
                xtrain = np.concatenate((xtrain, X_train), axis=0)

            if ytrain is None:
                ytrain = y_train
            else:
                ytrain = np.concatenate((ytrain, y_train), axis=0)

            if xtest is None:
                xtest = X_test
            else:
                xtest = np.concatenate((xtest, X_test), axis=0)

            if ytest is None:
                ytest = y_test
            else:
                ytest = np.concatenate((ytest, y_test), axis=0)

        train_indices = np.arange(len(xtrain))
        test_indices = np.arange(len(xtest))
        np.random.shuffle(train_indices)
        np.random.shuffle(test_indices)
        xtrain = xtrain[train_indices]
        ytrain = ytrain[train_indices]
        xtest = xtest[test_indices]
        ytest = ytest[test_indices]

        train_data.append({'x': xtrain, 'y': ytrain})
        num_samples['train'].append(len(ytrain))
        test_data.append({'x': xtest, 'y': ytest})
        num_samples['test'].append(len(ytest))
    
    print("Total number of samples:", sum(num_samples['train'] + num_samples['test']))
    print("The number of train samples:", num_samples['train'])
    print("The number of test samples:", num_samples['test'])

    del X, y
    return train_data, test_data

def save_file(config_path, train_path, test_path, train_data, test_data, num_clients, num_classes, statistic, alpha, batch_size, class_per_client, 
              train_ratio, least_samples, sampling_ratio, niid=False, balance=True, partition=None):
    import sys
    sys.exit()
    config = {
        'num_clients': num_clients, 
        'num_classes': num_classes, 
        'non_iid': niid, 
        'balance': balance, 
        'partition': partition, 
        'Size of samples for labels in clients': statistic, 
        'alpha': alpha, 
        'batch_size': batch_size, 
        'class_per_client': class_per_client,
        'train_ratio': train_ratio,
        'least_samples': least_samples,
        'sampling_ratio': sampling_ratio
    }

    print("Saving to disk.\n")

    for idx, train_dict in enumerate(train_data):
        with open(train_path + str(idx) + '.npz', 'wb') as f:
            np.savez_compressed(f, data=train_dict)
    for idx, test_dict in enumerate(test_data):
        with open(test_path + str(idx) + '.npz', 'wb') as f:
            np.savez_compressed(f, data=test_dict)
    with open(config_path, 'w') as f:
        ujson.dump(config, f)

    print("Finish generating dataset.\n")
