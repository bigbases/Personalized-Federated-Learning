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

import numpy as np
import pickle
import os
import sys
import random
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Subset
from utils.dataset_utils import check, separate_data, split_data, save_file,split_data_proportion

def subsample_dataset(dataset, samples_per_class, num_classes, emnist = False):
    if emnist:
        for i in range(len(dataset)):
            dataset.targets[i] = dataset.targets[i] - 1

    class_counts = {i: 0 for i in range(num_classes)}
    indices = []

    for i in range(len(dataset)):
        _, label = dataset[i]
        if class_counts[label] < samples_per_class:
            indices.append(i)
            class_counts[label] += 1
            if all(count == samples_per_class for count in class_counts.values()):
                break
    
    return torch.utils.data.Subset(dataset, indices)

def generate_mnist(dir_path, num_clients, num_classes, niid, balance, partition, class_per_client, train_ratio, 
                   alpha, batch_size, least_samples, sampling_ratio):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        
    # Setup directory for train/test data
    config_path = dir_path + "/config.json"
    train_path = dir_path + "/train/"
    test_path = dir_path + "/test/"

    if check(config_path, train_path, test_path):
        return

    # FIX HTTP Error 403: Forbidden
    from six.moves import urllib
    opener = urllib.request.build_opener()
    opener.addheaders = [('User-agent', 'Mozilla/5.0')]
    urllib.request.install_opener(opener)

    root_dir_path = dir_path.split("/")[0] + "/rawdata"

    # Get MNIST data

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])
    trainset = torchvision.datasets.MNIST(
        root=root_dir_path, train=True, download=True, transform=transform)
    testset = torchvision.datasets.MNIST(
        root=root_dir_path, train=False, download=True, transform=transform)
    
    trainset = subsample_dataset(trainset, (len(trainset)*sampling_ratio/num_classes), num_classes)
    testset = subsample_dataset(testset, (len(testset)*sampling_ratio/num_classes), num_classes)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=len(trainset), shuffle=False)
    testloader = torch.utils.data.DataLoader(testset, batch_size=len(testset), shuffle=False)

    dataset_image = []
    dataset_label = []

    for _, data in enumerate(trainloader, 0):
        inputs, labels = data
        dataset_image.extend(inputs.cpu().detach().numpy())
        dataset_label.extend(labels.cpu().detach().numpy())

    for _, data in enumerate(testloader, 0):
        inputs, labels = data
        dataset_image.extend(inputs.cpu().detach().numpy())
        dataset_label.extend(labels.cpu().detach().numpy())
    
    dataset_image = np.array(dataset_image)
    dataset_label = np.array(dataset_label)

    X, y, statistic = separate_data((dataset_image, dataset_label), num_clients, num_classes, alpha, 
                                    least_samples, niid, balance, partition, class_per_client)

    train_data, test_data = split_data_proportion(X, y, train_ratio)
    save_file(config_path, train_path, test_path, train_data, test_data, num_clients, num_classes, 
        statistic, alpha, batch_size, class_per_client, train_ratio, least_samples, sampling_ratio, niid, balance, partition)

def generate_fmnist(dir_path, num_clients, num_classes, niid, balance, partition, class_per_client, train_ratio, 
                   alpha, batch_size, least_samples, sampling_ratio):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        
    # Setup directory for train/test data
    config_path = dir_path + "/config.json"
    train_path = dir_path + "/train/"
    test_path = dir_path + "/test/"

    if check(config_path, train_path, test_path):
        return

    root_dir_path = dir_path.split("/")[0] + "/rawdata"

    # Get FashionMNIST data
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])

    trainset = torchvision.datasets.FashionMNIST(
        root=root_dir_path, train=True, download=True, transform=transform)
    testset = torchvision.datasets.FashionMNIST(
        root=root_dir_path, train=False, download=True, transform=transform)

    trainset = subsample_dataset(trainset, (len(trainset)*sampling_ratio/num_classes), num_classes)
    testset = subsample_dataset(testset, (len(testset)*sampling_ratio/num_classes), num_classes)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=len(trainset), shuffle=False)
    testloader = torch.utils.data.DataLoader(testset, batch_size=len(testset), shuffle=False)

    dataset_image = []
    dataset_label = []

    for _, data in enumerate(trainloader, 0):
        inputs, labels = data
        dataset_image.extend(inputs.cpu().detach().numpy())
        dataset_label.extend(labels.cpu().detach().numpy())

    for _, data in enumerate(testloader, 0):
        inputs, labels = data
        dataset_image.extend(inputs.cpu().detach().numpy())
        dataset_label.extend(labels.cpu().detach().numpy())
    

    dataset_image = np.array(dataset_image)
    dataset_label = np.array(dataset_label)



    X, y, statistic = separate_data((dataset_image, dataset_label), num_clients, num_classes, alpha, 
                                    least_samples, niid, balance, partition, class_per_client)


    train_data, test_data = split_data_proportion(X, y, train_ratio)
    save_file(config_path, train_path, test_path, train_data, test_data, num_clients, num_classes, 
        statistic, alpha, batch_size, class_per_client, train_ratio, least_samples, sampling_ratio, niid, balance, partition)
    
def generate_cifar10(dir_path, num_clients, num_classes, niid, balance, partition, class_per_client, train_ratio, 
                   alpha, batch_size, least_samples, sampling_ratio):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        
    config_path = dir_path + "/config.json"
    train_path = dir_path + "/train/"
    test_path = dir_path + "/test/"

    if check(config_path, train_path, test_path):
        return
    
    root_dir_path = dir_path.split("/")[0] + "/rawdata"

    # Get Cifar10 data
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR10(
        root=root_dir_path, train=True, download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10(
        root=root_dir_path, train=False, download=True, transform=transform)
    
    trainset = subsample_dataset(trainset, (len(trainset)*sampling_ratio/num_classes), num_classes)
    testset = subsample_dataset(testset, (len(testset)*sampling_ratio/num_classes), num_classes)


    trainloader = torch.utils.data.DataLoader(trainset, batch_size=len(trainset), shuffle=False)
    testloader = torch.utils.data.DataLoader(testset, batch_size=len(testset), shuffle=False)

    dataset_image = []
    dataset_label = []

    for _, data in enumerate(trainloader, 0):
        inputs, labels = data
        dataset_image.extend(inputs.cpu().detach().numpy())
        dataset_label.extend(labels.cpu().detach().numpy())

    for _, data in enumerate(testloader, 0):
        inputs, labels = data
        dataset_image.extend(inputs.cpu().detach().numpy())
        dataset_label.extend(labels.cpu().detach().numpy())
    
    # Convert lists to NumPy arrays
    dataset_image = np.array(dataset_image)
    dataset_label = np.array(dataset_label)

    X, y, statistic = separate_data((dataset_image, dataset_label), num_clients, num_classes, alpha,
                                    least_samples, niid, balance, partition, class_per_client)
    train_data, test_data = split_data_proportion(X, y, train_ratio)

    save_file(config_path, train_path, test_path, train_data, test_data, num_clients, num_classes, 
        statistic, alpha, batch_size, class_per_client, train_ratio, least_samples, sampling_ratio, niid, balance, partition)

def generate_cifar100(dir_path, num_clients, num_classes, niid, balance, partition, class_per_client, train_ratio, 
                   alpha, batch_size, least_samples, sampling_ratio):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        
    # Setup directory for train/test data
    config_path = dir_path + "/config.json"
    train_path = dir_path + "/train/"
    test_path = dir_path + "/test/"

    if check(config_path, train_path, test_path):
        return
    
    root_dir_path = dir_path.split("/")[0] + "/rawdata"

    # Get Cifar100 data
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR100(
        root=root_dir_path, train=True, download=True, transform=transform)
    testset = torchvision.datasets.CIFAR100(
        root=root_dir_path, train=False, download=True, transform=transform)
    
    trainset = subsample_dataset(trainset, (len(trainset)*sampling_ratio/num_classes), num_classes)
    testset = subsample_dataset(testset, (len(testset)*sampling_ratio/num_classes), num_classes)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=len(trainset), shuffle=False)
    testloader = torch.utils.data.DataLoader(testset, batch_size=len(testset), shuffle=False)

    dataset_image = []
    dataset_label = []

    for _, data in enumerate(trainloader, 0):
        inputs, labels = data
        dataset_image.extend(inputs.cpu().detach().numpy())
        dataset_label.extend(labels.cpu().detach().numpy())

    for _, data in enumerate(testloader, 0):
        inputs, labels = data
        dataset_image.extend(inputs.cpu().detach().numpy())
        dataset_label.extend(labels.cpu().detach().numpy())

    dataset_image = np.array(dataset_image)
    dataset_label = np.array(dataset_label)

    X, y, statistic = separate_data((dataset_image, dataset_label), num_clients, num_classes, alpha,
                                    least_samples, niid, balance, partition, class_per_client)

    train_data, test_data = split_data_proportion(X, y, train_ratio)
    save_file(config_path, train_path, test_path, train_data, test_data, num_clients, num_classes, 
        statistic, alpha, batch_size, class_per_client, train_ratio, least_samples, sampling_ratio, niid, balance, partition)

def generate_emnist(dir_path, num_clients, num_classes, niid, balance, partition, class_per_client, train_ratio, 
                   alpha, batch_size, least_samples, sampling_ratio):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        
    # Setup directory for train/test data
    config_path = dir_path + "/config.json"
    train_path = dir_path + "/train/"
    test_path = dir_path + "/test/"

    if check(config_path, train_path, test_path):
        return

    root_dir_path = dir_path.split("/")[0] + "/rawdata"

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])

    trainset = torchvision.datasets.EMNIST(
        root=root_dir_path, split='letters', train=True, download=True, transform=transform)
    testset = torchvision.datasets.EMNIST(
        root=root_dir_path, split='letters', train=False, download=True, transform=transform)

    trainset = subsample_dataset(trainset, (len(trainset)*sampling_ratio/num_classes), num_classes, True)
    testset = subsample_dataset(testset, (len(testset)*sampling_ratio/num_classes), num_classes, True)

    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=len(trainset), shuffle=False)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=len(testset), shuffle=False)

    dataset_image = []
    dataset_label = []

    for _, data in enumerate(trainloader, 0):
        inputs, labels = data
        dataset_image.extend(inputs.cpu().detach().numpy())
        dataset_label.extend(labels.cpu().detach().numpy())

    for _, data in enumerate(testloader, 0):
        inputs, labels = data
        dataset_image.extend(inputs.cpu().detach().numpy())
        dataset_label.extend(labels.cpu().detach().numpy())

    dataset_image = np.array(dataset_image)
    dataset_label = np.array(dataset_label)

    X, y, statistic = separate_data((dataset_image, dataset_label), num_clients, num_classes, alpha,
                                    least_samples, niid, balance, partition, class_per_client)

    train_data, test_data = split_data_proportion(X, y, train_ratio)
    save_file(config_path, train_path, test_path, train_data, test_data, num_clients, num_classes, 
        statistic, alpha, batch_size, class_per_client, train_ratio, least_samples, sampling_ratio, niid, balance, partition)

def generate_cinic10(dir_path, num_clients, num_classes, niid, balance, partition, class_per_client, train_ratio, 
                   alpha, batch_size, least_samples, sampling_ratio):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        
    # Setup directory for train/test data
    config_path = dir_path + "/config.json"
    train_path = dir_path + "/train/"
    test_path = dir_path + "/test/"

    if check(config_path, train_path, test_path):
        return
    
    cinic_mean = [0.47889522, 0.47227842, 0.43047404]
    cinic_std = [0.24205776, 0.23828046, 0.25874835]

    root_dir_path_train = dir_path.split("/")[0] + "/rawdata/train"
    root_dir_path_test = dir_path.split("/")[0] + "/rawdata/test"

    # Get Cinic10 data
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=cinic_mean, std=cinic_std)])

    trainset = torchvision.datasets.ImageFolder(
        root=root_dir_path_train, transform=transform)
    testset = torchvision.datasets.ImageFolder(
        root=root_dir_path_test, transform=transform)

    trainset = subsample_dataset(trainset, (len(trainset)*sampling_ratio/num_classes), num_classes)
    testset = subsample_dataset(testset, (len(testset)*sampling_ratio/num_classes), num_classes)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=len(trainset), shuffle=False)
    testloader = torch.utils.data.DataLoader(testset, batch_size=len(testset), shuffle=False)

    dataset_image = []
    dataset_label = []

    for _, data in enumerate(trainloader, 0):
        inputs, labels = data
        dataset_image.extend(inputs.cpu().detach().numpy())
        dataset_label.extend(labels.cpu().detach().numpy())

    for _, data in enumerate(testloader, 0):
        inputs, labels = data
        dataset_image.extend(inputs.cpu().detach().numpy())
        dataset_label.extend(labels.cpu().detach().numpy())
    
    dataset_image = np.array(dataset_image)
    dataset_label = np.array(dataset_label)

    X, y, statistic = separate_data((dataset_image, dataset_label), num_clients, num_classes, alpha,
                                    least_samples, niid, balance, partition, class_per_client)

    train_data, test_data = split_data_proportion(X, y, train_ratio)

    save_file(config_path, train_path, test_path, train_data, test_data, num_clients, num_classes, 
        statistic, alpha, batch_size, class_per_client, train_ratio, least_samples, sampling_ratio, niid, balance, partition)