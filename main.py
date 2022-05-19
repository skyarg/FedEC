import logging
import time
import json

import copy
import numpy as np
import torch
from utils.train_utils import get_data, get_model, read_data
from models.Update import LocalUpdate
from models.test import test_img_local_all

from log_utils.logger import args, info_logger
import os

save_dir = "save_" + args.alg + '_' + args.dataset + '_' + str(args.num_users) + '_' + str(args.shard_per_user) + "_" + str(args.attention)
log_file = './{}/info.log'.format(save_dir)

if not os.path.exists('./{}'.format(save_dir)):
    os.mkdir('./{}'.format(save_dir))
if not os.path.exists(log_file):
    with open(log_file, "w") as f:
        f.write("")

logging.basicConfig(filename=log_file, level=logging.DEBUG)

np.random.seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(args.seed)
torch.manual_seed(args.seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

if __name__ == '__main__':
    # parse args
    cuda0 = torch.device('cuda:' + str(args.gpu))
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    info_logger.info("Start experiment with args: \n {}".format(str(args)))
    lens = np.ones(args.num_users)

    # data processing
    if 'cifar' in args.dataset or args.dataset == 'mnist':
        dataset_train, dataset_test, dict_users_train, dict_users_test, user_label_dict = get_data(args)
        for idx in dict_users_train.keys():
            np.random.shuffle(dict_users_train[idx])
    else:
        train_path = './data/' + args.dataset + '/data/train'
        test_path = './data/' + args.dataset + '/data/test'

        clients, groups, dataset_train, dataset_test = read_data(train_path, test_path)
        lens = []
        for iii, c in enumerate(clients):
            lens.append(len(dataset_train[c]['x']))
        dict_users_train = list(dataset_train.keys())
        dict_users_test = list(dataset_test.keys())
        for c in dataset_train.keys():
            dataset_train[c]['y'] = list(np.asarray(dataset_train[c]['y']).astype('int64'))
            dataset_test[c]['y'] = list(np.asarray(dataset_test[c]['y']).astype('int64'))

    # build model
    net_glob = get_model(args)
    net_glob.train()

    # specify the representation parameters (in w_glob_keys) and head parameters (all others)
    if args.alg == 'fedec' or args.alg == 'fedrep' or args.alg == 'fedper':
        if 'cifar' in args.dataset:
            w_glob_keys = [net_glob.weight_keys[i] for i in [0, 1]]
        elif 'mnist' in args.dataset:
            w_glob_keys = [net_glob.weight_keys[i] for i in [0, 1, 2]]
        else:
            info_logger.info('Dataset {} in algorithm {} is not supported. '.format(args.dataset, args.alg))
    elif args.alg == 'lg':
        if 'cifar' in args.dataset:
            w_glob_keys = [net_glob.weight_keys[i] for i in [1, 2]]
        elif 'mnist' in args.dataset:
            w_glob_keys = [net_glob.weight_keys[i] for i in [2, 3]]
        else:
            info_logger.info('Dataset {} in algorithm {} is not supported. '.format(args.dataset, args.alg))

    w_locals = {}
    w_locals_with_global_para = {}
    for user in range(args.num_users):
        w_local_dict = {}
        for key in net_glob.state_dict().keys():
            w_local_dict[key] = net_glob.state_dict()[key]
        w_locals[user] = copy.deepcopy(w_local_dict)
        w_locals_with_global_para[user] = copy.deepcopy(w_local_dict)

    # training
    indd = None  # indices of embedding for sent140
    loss_train = []
    accs = []
    times = []
    accs10 = 0
    accs10_glob = 0
    start = time.time()

    acc_list = []
    acc_list_ = []
    next_sample_list = []
    next_sample_w_list = []

    for iter in range(args.epochs + 1):
        m = max(int(args.frac * args.num_users), 1)

        info_logger.info("epoch: \n {}".format(str(iter)))

        epoch_start = time.time()
        w_glob = {}
        loss_locals = []

        parameters = {}

        if iter == 0:
            idxs_users = np.random.choice(range(args.num_users), m, replace=False)
        else:
            idxs_users = next_sample_list
        info_logger.info("samples: \n {}".format(",".join([str(ele) for ele in idxs_users.tolist()])))

        w_keys_epoch = w_glob_keys
        times_in = []
        total_len = 0

        for ind, idx in enumerate(idxs_users):
            start_in = time.time()
            if 'femnist' in args.dataset in args.dataset:
                local = LocalUpdate(args=args, dataset=dataset_train[list(dataset_train.keys())[idx][:args.m_tr]],
                                                          idxs=dict_users_train, indd=indd)
            else:
                local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users_train[idx][:args.m_ft])

            net_local = copy.deepcopy(net_glob)
            teacher_net = copy.deepcopy(net_glob)#teacher 为前一轮的模型
            w_teacher = teacher_net.state_dict()
            for k in w_locals[idx].keys():
                w_teacher[k] = w_locals[idx][k]
            teacher_net.load_state_dict(w_teacher)

            # add temperature
            if iter == args.epochs:
                T = 1.0
            else:
                T= args.max_T - (iter+1) * (args.max_T-args.min_T) / args.epochs


            if 'femnist' in args.dataset:
                if args.is_decay:
                    if iter <=80:
                        args.lr = args.lr
                    elif iter <= 120:
                        args.lr = args.lr *0.995
                    else:
                        args.lr = args.lr * 0.99
                w_local, loss, indd = local.train(net=net_local.to(args.device), teacher_net=teacher_net,
                                                  w_glob_keys=w_glob_keys, T=T, lr=args.lr)
            else:
                w_local, loss, indd = local.train(net=net_local.to(args.device), teacher_net=teacher_net,
                                                  w_glob_keys=w_glob_keys, T=T, lr=args.lr)

            loss_locals.append(copy.deepcopy(loss))
            time_train_end = time.time()
            total_len += lens[idx]

            index = 0
            if len(w_glob) == 0:
                for k, key in enumerate(net_glob.state_dict().keys()):
                    if key in w_glob_keys:
                        if key not in parameters.keys():
                            parameters[key] = []
                        parameters[key].append(w_local[key])
                        index += 1
                    w_locals[idx][key] = copy.deepcopy(w_local[key])
                    w_locals_with_global_para[idx][key] = w_local[key]
            else:
                for k, key in enumerate(net_glob.state_dict().keys()):
                    if key in w_glob_keys:
                        if key not in parameters.keys():
                            parameters[key] = []
                        parameters[key].append(w_local[key])
                        # print("-<>-")
                        index += 1
                    w_locals[idx][key] = copy.deepcopy(w_local[key])
                    w_locals_with_global_para[idx][key] = w_local[key]
            times_in.append(time.time() - start_in)


        loss_avg = sum(loss_locals) / len(loss_locals)
        loss_train.append(loss_avg)


        for ind, idx in enumerate(idxs_users):
            if len(w_glob) == 0:
                w_glob = copy.deepcopy(w_locals[idx])
                for k, key in enumerate(net_glob.state_dict().keys()):
                    w_glob[key] = (w_glob[key] * lens[idx]).to(cuda0)
            else:
                for k, key in enumerate(net_glob.state_dict().keys()):
                    if key in w_glob_keys:
                        w_glob[key] += (w_locals[idx][key] * lens[idx]).to(cuda0)
                    else:
                        w_glob[key] += (w_locals[idx][key] * lens[idx]).to(cuda0)
        # get weighted average for global weights
        for k in net_glob.state_dict().keys():
            w_glob[k] = torch.div(w_glob[k], total_len)
        net_glob.load_state_dict(w_glob)

        next_sample_list = np.random.choice(range(args.num_users), m, replace=False)

        # testing
        if iter % args.test_freq == args.test_freq - 1:

            if times == []:
                times.append(max(times_in))
            else:
                times.append(times[-1] + max(times_in))
            acc_test, loss_test = test_img_local_all(net_glob, args, dataset_test, dict_users_test, sampled_client_list=idxs_users,
                                                     w_glob_keys=w_glob_keys, w_locals=w_locals, indd=indd,
                                                     dataset_train=dataset_train, dict_users_train=dict_users_train,
                                                     return_all=False, iter=iter)
            accs.append(acc_test)
            info_logger.info('Round {:3d}, Train loss: {:.3f}, Test loss: {:.3f}, Test accuracy: {:.2f}'.format(
                    iter, loss_avg, loss_test, acc_test))

        # saving
        if iter % args.save_every == args.save_every - 1:
            model_save_path = './save/accs_' + args.alg + '_' + args.dataset + '_' + str(args.num_users) + '_' + str(
                args.shard_per_user) + '_iter' + str(iter) + args.function + '.pt'
            torch.save(net_glob.state_dict(), model_save_path)
        info_logger.info("each epoch  cost time:%s" % str(time.time() - epoch_start))
