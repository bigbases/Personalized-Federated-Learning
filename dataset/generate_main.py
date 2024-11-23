import argparse
from generators.generators import generate_mnist, generate_fmnist, generate_cifar10, generate_cifar100, generate_emnist, generate_cinic10
import numpy as np
import random
seed = 5
np.random.seed(seed)
random.seed(seed)

def run(args):
    if args.data_dist == 'niid':
        niid = True
    else:
        niid = False
    if args.balance == 'balance':
        balance = True
    else:
        balance = False
    if args.dataset == 'mnist':
        generate_mnist(args.dir_path, args.num_clients, args.num_classes, niid, balance, args.partition, args.class_per_client, args.train_ratio,
                       args.alpha, args.batch_size, args.least_samples, args.sampling_ratio)
    elif args.dataset == 'fmnist':
        generate_fmnist(args.dir_path, args.num_clients, args.num_classes, niid, balance, args.partition, args.class_per_client, args.train_ratio,
                       args.alpha, args.batch_size, args.least_samples, args.sampling_ratio)
    elif args.dataset == 'cifar10':
        generate_cifar10(args.dir_path, args.num_clients, args.num_classes, niid, balance, args.partition, args.class_per_client, args.train_ratio,
                       args.alpha, args.batch_size, args.least_samples, args.sampling_ratio)
    elif args.dataset == 'cifar100':
        generate_cifar100(args.dir_path, args.num_clients, args.num_classes, niid, balance, args.partition, args.class_per_client, args.train_ratio,
                       args.alpha, args.batch_size, args.least_samples, args.sampling_ratio)
    elif args.dataset in ['cinic10', 'cinic10-imagenet']:
        generate_cinic10(args.dir_path, args.num_clients, args.num_classes, niid, balance, args.partition, args.class_per_client, args.train_ratio,
                       args.alpha, args.batch_size, args.least_samples, args.sampling_ratio)
        
    else:
        print("Dataset not found!")
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', "--dataset", type=str, default="mnist", 
                        help="The name of dataset", choices=['mnist', 'fmnist', 'cifar10', 'cifar100', 'cinic10','cinic10-imagenet'])
    parser.add_argument('-nclasses', "--num_classes", type=int, default=10,
                        help="The number of classes")
    parser.add_argument('-dir', '--dir_path', type=str, default="dataset/",
                        help="The path of dataset")
    parser.add_argument('-niid', '--data_dist', type=str, default='niid',
                        help="The data distribution is non-iid or not", choices=['iid', 'niid'])
    parser.add_argument('-bal', "--balance", type=str, default='balance',
                        help="The data distribution is balanced or not", choices=['balance', 'unbalance'])
    parser.add_argument('-par', "--partition", type=str, default=None,
                        help="The partition of data")
    parser.add_argument('-nclients', "--num_clients", type=int, default=20,
                        help="The number of clients")
    parser.add_argument('-cpc', "--class_per_client", type=int, default=2)
    parser.add_argument('-tr', "--train_ratio", type=float, default=0.75)
    parser.add_argument('-a', "--alpha", type=float, default=0.1)
    parser.add_argument('-bs', "--batch_size", type=int, default=32)
    parser.add_argument('-ls', "--least_samples", type=int, default=1)
    parser.add_argument('-spr', "--sampling_ratio", type=float, default=1,
                        help='Number of classes per client relative to overall size (e.g, 0.5 of MNIST -> 70000 * 0.5 = 35000 / 20) -> 1750')
    args = parser.parse_args()
    run(args)


