# Modified from: https://github.com/pliang279/LG-FedAvg/blob/master/models/Update.py

#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import numpy as np

class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs, name=None, pseudo_label=None):
        self.dataset = dataset
        self.idxs = list(idxs)
        self.name = name
        self.pseudo_label = pseudo_label

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        if self.name is None:
            if self.pseudo_label is None:
                image, label = self.dataset[self.idxs[item]]
            else:
                image, label = self.dataset[self.idxs[item]][0], self.pseudo_label[item]
        elif 'femnist' in self.name:
            image = torch.reshape(torch.tensor(self.dataset['x'][item]),(1,28,28))
            label = torch.tensor(self.dataset['y'][item])
        elif 'sent140' in self.name:
            image = self.dataset['x'][item]
            label = self.dataset['y'][item]
        else:
            image, label = self.dataset[self.idxs[item]]
        return image, label

class LocalUpdate(object):
    def __init__(self, args, dataset=None, idxs=None):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()

        if 'femnist' in args.dataset in args.dataset:
            self.ldr_train = DataLoader(DatasetSplit(dataset, np.ones(len(dataset['x'])), name=self.args.dataset),
                                        batch_size=self.args.local_bs, shuffle=True)
        else:
            self.ldr_train = DataLoader(DatasetSplit(dataset, idxs), batch_size=self.args.local_bs, shuffle=True)

        self.dataset = dataset
        self.idxs = idxs

    def loss_func_KL(self, p_logit, q_logit):
        p = F.softmax(p_logit, dim=-1)
        _kl = torch.sum(p * (F.log_softmax(p_logit, dim=-1) - F.log_softmax(q_logit, dim=-1)), 1)
        return torch.mean(_kl)

    def train(self, net, teacher_net, T, lr=0.1):
        bias_p = []
        weight_p = []
        for name, p in net.named_parameters():
            if 'bias' in name:
                bias_p += [p]
            else:
                weight_p += [p]
        optimizer = torch.optim.SGD(
            [
                {'params': weight_p, 'weight_decay': 0.0001},
                {'params': bias_p, 'weight_decay': 0}
            ],
            lr=lr, momentum=0.5
        )

        local_eps = self.args.local_ep

        for name, param in net.named_parameters():
            param.requires_grad = True

        epoch_loss = []

        teacher_params = {n: p.clone().detach() for n, p in teacher_net.named_parameters()}  # 模型的所有参数

        for iter in range(local_eps):

            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(self.args.device), labels.to(self.args.device)

                middle_output,teacher_labels = teacher_net(images, T)
                net.zero_grad()
                student_output,log_probs = net(images)
                loss = self.loss_func(log_probs, labels)
                if self.args.l2_type == 'l2':
                    # l2 loss
                    l2_loss = 0
                    for n, p in net.named_parameters():
                        _loss = (p - teacher_params[n]) ** 2
                        l2_loss += _loss.sum()
                    loss += self.args.l2_weight * l2_loss

                else:
                    dim_labels = labels.size(-1)
                    dim_middle_labels = middle_output.size(-1)
                    dim_sum = dim_labels + dim_middle_labels
                    dim_labels /= dim_sum
                    dim_middle_labels /= dim_sum
                    loss += self.args.KL_label_weight * dim_labels * self.loss_func_KL(log_probs, teacher_labels) + \
                            self.args.KL_feature_weight * dim_middle_labels * self.loss_func_KL(middle_output,
                                                                                                student_output)


                loss.backward()
                optimizer.step()

            epoch_loss.append(sum(batch_loss) / len(batch_loss))

        return net.state_dict(), sum(epoch_loss) / len(epoch_loss)