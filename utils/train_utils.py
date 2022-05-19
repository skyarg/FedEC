# Modified from: https://github.com/pliang279/LG-FedAvg/blob/master/utils/train_utils.py

from torchvision import datasets, transforms
from models.Nets import MLP, CNNCifar100Multi, CNNCifarMulti, MLPMulti, CNN_FEMNISTMulti
from utils.sampling import noniid, noniid_global
import os
import json
from log_utils.logger import info_logger

trans_mnist = transforms.Compose([transforms.ToTensor(),
                                  transforms.Normalize((0.1307,), (0.3081,))])
trans_cifar10_train = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(),
                                          transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                               std=[0.229, 0.224, 0.225])])
trans_cifar10_val = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                             std=[0.229, 0.224, 0.225])])
trans_cifar100_train = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(),
                                          transforms.Normalize(mean=[0.507, 0.487, 0.441],
                                                               std=[0.267, 0.256, 0.276])])
trans_cifar100_val = transforms.Compose([transforms.ToTensor(),
                                         transforms.Normalize(mean=[0.507, 0.487, 0.441],
                                                              std=[0.267, 0.256, 0.276])])


def get_data(args):
    if args.dataset == 'mnist':
        dataset_train = datasets.MNIST('./data/mnist/', train=True, download=True, transform=trans_mnist)
        dataset_test = datasets.MNIST('./data/mnist/', train=False, download=True, transform=trans_mnist)
        dict_users_train, rand_set_all = noniid(dataset_train, args.num_users, args.shard_per_user,
                                                                     args.num_classes)
        dict_users_test, rand_set_all = noniid(dataset_test, args.num_users, args.shard_per_user, args.num_classes, rand_set_all=rand_set_all, testb=True)
    elif args.dataset == 'cifar10':
        dataset_train = datasets.CIFAR10('./data/cifar10', train=True, download=True, transform=trans_cifar10_train)
        dataset_test = datasets.CIFAR10('./data/cifar10', train=False, download=True, transform=trans_cifar10_val)
        dict_users_train, rand_set_all, global_train = noniid_global(dataset_train, args.num_users, args.shard_per_user, args.num_classes)
        dict_users_test, rand_set_all = noniid(dataset_test, args.num_users, args.shard_per_user, args.num_classes, rand_set_all=rand_set_all, testb=True)
    elif args.dataset == 'cifar100':
        dataset_train = datasets.CIFAR100('./data/cifar100', train=True, download=True, transform=trans_cifar100_train)
        dataset_test = datasets.CIFAR100('./data/cifar100', train=False, download=True, transform=trans_cifar100_val)
        dict_users_train, rand_set_all, global_train = noniid_global(dataset_train, args.num_users, args.shard_per_user,
                                                                     args.num_classes)
        dict_users_test, rand_set_all = noniid(dataset_test, args.num_users, args.shard_per_user, args.num_classes, rand_set_all=rand_set_all, testb=True)
    else:
        exit('Error: unrecognized dataset')
    temp = {i: list(tmp) for i, tmp in enumerate(rand_set_all)}
    info_logger.info("rand_set_all: \n{}".format(str(temp)))
    return dataset_train, dataset_test, dict_users_train, dict_users_test, temp
    # return dataset_train, dataset_test, dict_users_train, dict_users_test, global_train

def read_data(train_data_dir, test_data_dir):
    clients = []
    groups = []
    train_data = {}
    test_data = {}

    train_files = os.listdir(train_data_dir)
    train_files = [f for f in train_files if f.endswith('.json')]
    for f in train_files:
        file_path = os.path.join(train_data_dir,f)
        with open(file_path, 'r') as inf:
            cdata = json.load(inf)
        clients.extend(cdata['users'])
        if 'hierarchies' in cdata:
            groups.extend(cdata['hierarchies'])
        train_data.update(cdata['user_data'])

    test_files = os.listdir(test_data_dir)
    test_files = [f for f in test_files if f.endswith('.json')]
    for f in test_files:
        file_path = os.path.join(test_data_dir,f)
        with open(file_path, 'r') as inf:
            cdata = json.load(inf)
        test_data.update(cdata['user_data'])

    clients = list(train_data.keys())

    return clients, groups, train_data, test_data


def get_model(args):
    if args.model == 'cnn' and 'cifar100' in args.dataset:
        net_glob = CNNCifar100Multi(args=args).to(args.device)
    elif args.model == 'cnn' and 'cifar10' in args.dataset:
        net_glob = CNNCifarMulti(args=args).to(args.device)
    elif args.model == 'mlp' and 'mnist' in args.dataset:
        net_glob = MLPMulti(dim_in=784, dim_hidden=256, dim_out=args.num_classes).to(args.device)
    elif args.model == 'cnn' and 'femnist' in args.dataset:
        net_glob = CNN_FEMNISTMulti(args=args).to(args.device)
    elif args.model == 'mlp' and 'cifar' in args.dataset:
        net_glob = MLP(dim_in=3072, dim_hidden=512, dim_out=args.num_classes).to(args.device)
    else:
        exit('Error: unrecognized model')

    return net_glob