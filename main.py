#!/usr/bin/env python
import copy
import torch
import argparse
import os
import time
import warnings
import numpy as np
import torchvision
import logging

from server.serversdp import FedSDP
from model.resnetsdp import *

from utils.result_utils import average_data
from utils.mem_utils import MemReporter

logger = logging.getLogger()
logger.setLevel(logging.ERROR)

warnings.simplefilter("ignore")
torch.manual_seed(0)


def run(args):
    time_list = []
    model_str = args.model

    for i in range(args.prev, args.times):
        print(f"\n============= Running time: {i}th =============")
        print("Creating server and clients ...")
        start = time.time()

        if model_str == "resnetfc2":
            if "cifar10" in args.dataset:
                args.model = resnet18fc2(channels=3, num_classes=args.num_classes, has_bn=True, bn_block_num=4).to(args.device)
            elif "cinic10" in args.dataset:
                args.model = resnet18fc2(channels=3, num_classes=args.num_classes, has_bn=True, bn_block_num=4).to(args.device)
            elif "cinic10-imagenet" in args.dataset:
                args.model = resnet18fc2(channels=3, num_classes=args.num_classes, has_bn=True, bn_block_num=4).to(args.device)
            elif "cifar100" in args.dataset:
                args.model = resnet18fc2(channels=3, num_classes=args.num_classes, has_bn=True, bn_block_num=4).to(args.device)
            else:
                args.model = resnet18fc2(channels=1, num_classes=args.num_classes, has_bn=True, bn_block_num=4).to(args.device)

        else:
            raise NotImplementedError

        print(args.model)

        if args.algorithm == "FedSDP":
            args.model_name = model_str
            args.per = copy.deepcopy(args.model.fc1)
            args.head = copy.deepcopy(args.model.fc2)
            args.model.fc1 = nn.Identity()
            args.model.fc2 = nn.Identity()
            args.model = BaseHeadSplitProto(args.model, args.per, args.head)
            server = FedSDP(args, i)

        else:
            raise NotImplementedError

        server.train(i)

        time_list.append(time.time()-start)

    print(f"\nAverage time cost: {round(np.average(time_list), 2)}s.")
    print("All done!")
    
if __name__ == "__main__":
    total_start = time.time()

    parser = argparse.ArgumentParser()
    parser.add_argument('-go', "--goal", type=str, default="test", 
                        help="The goal for this experiment")
    parser.add_argument('-dev', "--device", type=str, default="cuda",
                        choices=["cpu", "cuda"])
    parser.add_argument('-did', "--device_id", type=str, default="2")
    parser.add_argument('-data', "--dataset", type=str, default="mnist")
    parser.add_argument('-nb', "--num_classes", type=int, default=10)
    parser.add_argument('-m', "--model", type=str, default="cnn")
    parser.add_argument('-lbs', "--batch_size", type=int, default=10)
    parser.add_argument('-lr', "--local_learning_rate", type=float, default=0.005,
                        help="Local learning rate")
    parser.add_argument('-ld', "--learning_rate_decay", type=bool, default=False)
    parser.add_argument('-ldg', "--learning_rate_decay_gamma", type=float, default=0.99)
    parser.add_argument('-gr', "--global_rounds", type=int, default=2000)
    parser.add_argument('-ls', "--local_epochs", type=int, default=1, 
                        help="Multiple update steps in one local epoch.")
    parser.add_argument('-algo', "--algorithm", type=str, default="FedAvg")
    parser.add_argument('-jr', "--join_ratio", type=float, default=1.0,
                        help="Ratio of clients per round")
    parser.add_argument('-rjr', "--random_join_ratio", type=bool, default=False,
                        help="Random ratio of clients per round")
    parser.add_argument('-nc', "--num_clients", type=int, default=20,
                        help="Total number of clients")
    parser.add_argument('-pv', "--prev", type=int, default=0,
                        help="Previous Running times")
    parser.add_argument('-t', "--times", type=int, default=1,
                        help="Running times")
    parser.add_argument('-eg', "--eval_gap", type=int, default=1,
                        help="Rounds gap for evaluation")
    parser.add_argument('-dp', "--privacy", type=bool, default=False,
                        help="differential privacy")
    parser.add_argument('-dps', "--dp_sigma", type=float, default=0.0)
    parser.add_argument('-sfn', "--save_folder_name", type=str, default='items')
    parser.add_argument('-ab', "--auto_break", type=bool, default=False)
    parser.add_argument('-dlg', "--dlg_eval", type=bool, default=False)
    parser.add_argument('-dlgg', "--dlg_gap", type=int, default=100)
    parser.add_argument('-bnpc', "--batch_num_per_client", type=int, default=2)
    parser.add_argument('-nnc', "--num_new_clients", type=int, default=0)
    parser.add_argument('-ften', "--fine_tuning_epoch_new", type=int, default=0)
    parser.add_argument('-cdr', "--client_drop_rate", type=float, default=0.0,
                        help="Rate for clients that train but drop out")
    parser.add_argument('-tsr', "--train_slow_rate", type=float, default=0.0,
                        help="The rate for slow clients when training locally")
    parser.add_argument('-ssr', "--send_slow_rate", type=float, default=0.0,
                        help="The rate for slow clients when sending global model")
    parser.add_argument('-ts', "--time_select", type=bool, default=False,
                        help="Whether to group and select clients at each round according to time cost")
    parser.add_argument('-tth', "--time_threthold", type=float, default=10000,
                        help="The threthold for droping slow clients")

    parser.add_argument('-dir', '--data_dir', type=str, default='data')
    parser.add_argument('-mo', '--momentum', type=float, default=0.5)
    parser.add_argument('-wd', '--weight_decay', type=float, default=1e-4)
    parser.add_argument('-optim', '--optimizer', type=str, default='sgd')
    parser.add_argument('-tw', "--test_way", type=str, default='ind', choices=['ind','agg'])
    parser.add_argument('-ls1', "--local_epoch1", type=int, default=1)
    parser.add_argument('-ls2', "--local_epoch2", type=int, default=1)
    parser.add_argument('-ls3', "--local_epoch3", type=int, default=1)
    parser.add_argument('-mo1', "--momentum1", type=float, default=0.5)
    parser.add_argument('-mo2', "--momentum2", type=float, default=0.5)
    parser.add_argument('-mo3', "--momentum3", type=float, default=0.5)
    parser.add_argument('-lda', "--lamda", type=float, default=0.1)
    
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.device_id

    if args.device == "cuda" and not torch.cuda.is_available():
        print("\ncuda is not avaiable.\n")
        args.device = "cpu"

    run(args)


