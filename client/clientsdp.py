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

import torch
import copy
import numpy as np
import torch.nn as nn
import time
from .clientbase import Client
from collections import defaultdict
import torch.nn.functional as F
import pickle
import random
import os


class clientSDP(Client):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)

        self.personalized_protos_grep = None
        self.personalized_protos_rep = None
        
        self.local_epoch1 = args.local_epoch1
        self.local_epoch2 = args.local_epoch2
        self.local_epoch3 = args.local_epoch3

        self.momentum1 = args.momentum1
        self.momentum2 = args.momentum2
        self.momentum3 = args.momentum3

        self.unique_seed = 0
        self.goal = args.goal

        self.norm_base = []
        self.current_params_base = None
        self.previous_params_base = None
        self.running_min = []

        self.loss_mse = nn.MSELoss()
        self.lda = args.lamda
        self.next_lambda = 1
        self.lambdas = None

        self.optimizer_body = torch.optim.SGD(self.model.base.parameters(), lr=self.learning_rate, momentum=self.momentum1)
        self.learning_rate_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer=self.optimizer_body, 
            gamma=args.learning_rate_decay_gamma
        )

        self.optimizer_per = torch.optim.SGD(self.model.per.parameters(), lr=self.learning_rate, momentum=self.momentum2)
        self.learning_rate_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer=self.optimizer_per, 
            gamma=args.learning_rate_decay_gamma
        )

        self.optimizer_head = torch.optim.SGD(self.model.head.parameters(), lr=self.learning_rate, momentum=self.momentum3)
        self.learning_rate_scheduler_head = torch.optim.lr_scheduler.ExponentialLR(
            optimizer=self.optimizer_head, 
            gamma=args.learning_rate_decay_gamma
        )
    
        self.bar_C = []
        self.C = []


    def train(self, i):
        seed = self.train_time_cost['num_rounds']
        self.unique_seed = seed + (i * 100)
        trainloader = self.load_train_data()
        start_time = time.time()
        self.model.train()

        # Classification Stage
        self.personalized_protos_rep, general_protos = train_head(self.unique_seed, self.model, trainloader, self.local_epoch1, self.device, self.loss, self.optimizer_head)
        self.bar_C.append(general_protos)

        # Prototyping Stage
        if self.next_lambda != 0:
            self.personalized_protos_grep = train_personal(self.unique_seed, self.model, trainloader, self.local_epoch2, self.device, self.loss, self.loss_mse, 
                                                       self.optimizer_per, self.personalized_protos_rep, self.next_lambda, self.train_time_cost, self.lambdas)
            self.lambdas = {key: 0 for key in self.personalized_protos_grep.keys()}
        else:
            self.personalized_protos_grep = None
            self.lambdas = None

        self.C.append(self.personalized_protos_grep)

        # Weight Calculation
        if (self.next_lambda != 0) and (self.train_time_cost['num_rounds'] != 0):
            for cls in self.personalized_protos_grep.keys():
                bar_C_tensor = general_protos[cls]
                C_tensor = self.personalized_protos_grep[cls]
                if type(bar_C_tensor) is list:
                    if type(self.bar_C[-2][cls]) is list:
                        bar_C_tensor = C_tensor
                    else:
                        bar_C_tensor = self.bar_C[-2][cls]
                if type(C_tensor) is list:
                    if type(self.C[-2][cls] is list):
                        C_tensor = bar_C_tensor
                    else:
                        C_tensor = self.C[-2][cls]
                
                diff = bar_C_tensor - C_tensor
                
                l2 = torch.mean(diff**2).item() / self.lda

                if (l2 > 1) or (l2 == 0):
                    self.lambdas[cls] = 1
                else:
                    self.lambdas[cls] = l2

        # Localization Stage
        self.previous_params_base = get_base_params_tensor(self.model)
        _, general_protos, _, _, self.next_lambda = train_body(self.unique_seed, self.model, trainloader, self.local_epoch3, self.device, 
                                                                                                                 self.loss, self.loss_mse, self.optimizer_body, self.personalized_protos_grep, self.next_lambda, self.train_time_cost, self.lambdas)
        self.current_params_base = get_base_params_tensor(self.model)
        params_diff_base = torch.norm(self.current_params_base - self.previous_params_base).item()
        self.norm_base.append(params_diff_base)

        # Set I
        if self.next_lambda != 0:
            ind, _ = find_overall_minimum(self.norm_base)
            self.running_min.append(ind)
            if self.running_min.count(self.running_min[-1]) == 3:
                self.next_lambda = 0

        if self.learning_rate_decay:
            self.learning_rate_scheduler.step()
            self.learning_rate_scheduler_per.step()

        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time

    def set_parameters(self, base):
        for new_param, old_param in zip(base.parameters(), self.model.base.parameters()):
            old_param.data = new_param.data.clone()

    def train_metrics(self):
        torch.manual_seed(self.unique_seed)
        
        trainloader = self.load_train_data()
        self.model.eval()

        proto_loss_body = 0
        proto_loss_per = 0
        train_num = 0
        losses = 0
        with torch.no_grad():
            for x, y in trainloader:
                x = x[0].to(self.device) if isinstance(x, list) else x.to(self.device)
                y = y.to(self.device)
                general_rep = self.model.base(x)
                rep = self.model.per(general_rep)
                output = self.model.head(rep)
                loss = self.loss(output, y)

                if self.personalized_protos_rep is not None:
                    proto_new = copy.deepcopy(rep.detach())
                    for i, yy in enumerate(y):
                        y_c = yy.item()
                        if type(self.personalized_protos_rep[y_c]) != type([]):
                            proto_new[i, :] = self.personalized_protos_rep[y_c].data

                    proto_loss_per += self.loss_mse(proto_new, rep) * y.shape[0]

                if self.personalized_protos_grep is not None:
                    proto_new = copy.deepcopy(rep.detach())
                    for i, yy in enumerate(y):
                        y_c = yy.item()
                        if type(self.personalized_protos_grep[y_c]) != type([]):
                            proto_new[i, :] = self.personalized_protos_grep[y_c].data

                    proto_loss_body += self.loss_mse(proto_new, general_rep) * y.shape[0]

                train_num += y.shape[0]
                losses += loss.item() * y.shape[0]

        if self.personalized_protos_grep is not None:
            proto_loss_body = proto_loss_body.cpu().numpy().item()
        if self.personalized_protos_rep is not None:
            proto_loss_per = proto_loss_per.cpu().numpy().item()

        return losses, train_num, proto_loss_body, proto_loss_per


def collect_protos(model, device, trainloader, personal = True):
    copymodel = copy.deepcopy(model)
    copymodel.to(device)
    copymodel.eval()

    protos = defaultdict(list)
    with torch.no_grad():
        for i, (x, y) in enumerate(trainloader):
            if type(x) == type([]):
                x[0] = x[0].to(device)
            else:
                x = x.to(device)
            y = y.to(device)
            general_rep = copymodel.base(x)
            rep = copymodel.per(general_rep)

            if personal:
                for i, yy in enumerate(y):
                    y_c = yy.item()
                    protos[y_c].append(rep[i, :].detach().data)
            else:
                for i, yy in enumerate(y):
                    y_c = yy.item()
                    protos[y_c].append(general_rep[i, :].detach().data)
    return protos
        
def train_head(seed, model, trainloader, local_epoch, device, loss_ce, optimizer):
    torch.manual_seed(seed)
    for param in model.base.parameters():
        param.requires_grad = False
    for param in model.per.parameters():
        param.requires_grad = False
    for param in model.head.parameters():
        param.requires_grad = True

    personalized_protos = defaultdict(list)
    general_protos = defaultdict(list)
    for epoch in range(local_epoch):
        for i, (x, y) in enumerate(trainloader):
            x = x[0].to(device) if isinstance(x, list) else x.to(device)
            y = y.to(device)
            general_rep = model.base(x)
            rep = model.per(general_rep)
            output = model.head(rep)
            loss = loss_ce(output, y)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    personalized_protos = agg_func(collect_protos(model, device, trainloader, personal=True))
    general_protos = agg_func(collect_protos(model, device, trainloader, personal=False))

    return personalized_protos, general_protos

def train_personal(seed, model, trainloader, local_epoch, device, loss_ce, loss_mse, optimizer, personalized_protos_rep, lamda, train_time_cost, lambdas):
    torch.manual_seed(seed)

    gr = train_time_cost['num_rounds']

    for param in model.base.parameters():
        param.requires_grad = False
    for param in model.per.parameters():
        param.requires_grad = True
    for param in model.head.parameters():
        param.requires_grad = False

    personalized_protos = defaultdict(list)
    for epoch in range(local_epoch):
        for i, (x, y) in enumerate(trainloader):
            x = x[0].to(device) if isinstance(x, list) else x.to(device)
            y = y.to(device)
            general_rep = model.base(x)
            rep = model.per(general_rep)

            proto_new = copy.deepcopy(rep.detach())
            weights = []
            for i, yy in enumerate(y):
                y_c = yy.item()
                if type(personalized_protos_rep[y_c]) != type([]):
                    proto_new[i, :] = personalized_protos_rep[y_c].data

                if lambdas is not None:
                    try:
                        weights.append(lambdas[y_c])
                    except KeyError:
                        weights.append(1)

            if gr > 1:
                weights = torch.tensor(weights, dtype=torch.float32).view(-1, 1).expand_as(proto_new).to(device)

                squared_diff = (proto_new - rep) ** 2
                weighted_squared_diff = weights * squared_diff
                loss = weighted_squared_diff.sum() / (proto_new.size(0) * proto_new.size(1)) * lamda

            else:
                loss = loss_mse(proto_new, rep) * lamda

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    personalized_protos = agg_func(collect_protos(model, device, trainloader, personal=True))

    return personalized_protos

def train_body(seed, model, trainloader, local_epoch, device, loss_ce, loss_mse, optimizer, personalized_protos_grep, lamda, train_time_cost, lambdas):
        
    torch.manual_seed(seed)
    for param in model.base.parameters():
        param.requires_grad = True
    for param in model.per.parameters():
        param.requires_grad = False
    for param in model.head.parameters():
        param.requires_grad = False

    losses_mse = 0
    losses_ce = 0
    train_num = 0

    personalized_protos = defaultdict(list)
    general_protos = defaultdict(list)
    gr = train_time_cost['num_rounds']
    for epoch in range(local_epoch):
        for i, (x, y) in enumerate(trainloader):
            x = x[0].to(device) if isinstance(x, list) else x.to(device)
            y = y.to(device)
            general_rep = model.base(x)
            rep = model.per(general_rep)
            output = model.head(rep)

            loss = loss_ce(output, y)
            losses_ce += loss.item() * y.shape[0]
            train_num += y.shape[0]

            if personalized_protos_grep is not None:
                proto_new = copy.deepcopy(rep.detach())
                weights = []
                for i, yy in enumerate(y):
                    y_c = yy.item()
                    if type(personalized_protos_grep[y_c]) != type([]):
                        proto_new[i, :] = personalized_protos_grep[y_c].data
                    if lambdas is not None:
                        try:
                            weights.append(lambdas[y_c])
                        except KeyError:
                            weights.append(1)
                
                if gr != 0:
                    weights = torch.tensor(weights, dtype=torch.float32).view(-1, 1).expand_as(proto_new).to(device)

                    squared_diff = (proto_new - general_rep) ** 2
                    weighted_squared_diff = weights * squared_diff
                    lossmse = weighted_squared_diff.sum() / (proto_new.size(0) * proto_new.size(1))

                else:
                    lossmse = loss_mse(proto_new, general_rep)

                loss += lossmse * lamda
                losses_mse += lossmse.item() * y.shape[0]

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    if lamda != 0:
        next_lambda = 1
    else:
        next_lambda = 0

    personalized_protos = agg_func(collect_protos(model, device, trainloader, personal=True))
    general_protos = agg_func(collect_protos(model, device, trainloader, personal=False))
    return personalized_protos, general_protos, losses_ce/train_num, losses_mse/train_num, next_lambda

def get_base_params_tensor(model):
    return torch.cat([p.data.view(-1) for p in model.base.parameters()])

def agg_func(protos):
    """
    Returns the average of the weights.
    """

    for [label, proto_list] in protos.items():
        if len(proto_list) > 1:
            proto = 0 * proto_list[0].data
            for i in proto_list:
                proto += i.data
            protos[label] = proto / len(proto_list)
        else:
            protos[label] = proto_list[0]

    return protos

def find_overall_minimum(losses):
    min_value = float('inf')
    min_index = -1
    for i, loss in enumerate(losses):
        if loss < min_value:
            min_value = loss
            min_index = i
    return min_index, min_value
