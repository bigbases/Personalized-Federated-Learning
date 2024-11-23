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
import torch.nn as nn
import h5py
import os
import random
import time
import numpy as np
from client.clientsdp import clientSDP
from .serverbase import Server
from threading import Thread
from collections import defaultdict
import pickle


class FedSDP(Server):
    def __init__(self, args, times):
        super().__init__(args, times)

        self.set_slow_clients()
        self.set_clients(clientSDP)

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")

        self.Budget = []

        self.losses_b_by_clients = defaultdict(list)
        self.losses_p_by_clients = defaultdict(list)

        self.current_params = None
        self.previous_params = None

        self.norm_hist = []

    def train(self, j):
        for i in range(self.global_rounds+1):
            s_t = time.time()
            self.selected_clients = self.select_clients()
            self.send_models()

            if i%self.eval_gap == 0:
                print(f"\n-------------Round number: {i}-------------")
                print("\nEvaluate global models")
                self.evaluate()

            for client in self.selected_clients:
                client.train(j)

            self.receive_models()
            if self.dlg_eval and i%self.dlg_gap == 0:
                self.call_dlg(i)
            self.aggregate_parameters()

            self.Budget.append(time.time() - s_t)
            print('-'*25, 'time cost', '-'*25, self.Budget[-1])

            if self.auto_break and self.check_done(acc_lss=[self.rs_test_acc], top_cnt=self.top_cnt):
                break

        print("\nBest accuracy.")
        print(max(self.rs_test_acc))
        print("\nAverage time cost per round.")
        print(sum(self.Budget[1:])/len(self.Budget[1:]))

        self.save_results()


    def receive_models(self):
        assert (len(self.selected_clients) > 0)

        active_clients = random.sample(
            self.selected_clients, int((1-self.client_drop_rate) * self.current_num_join_clients))

        self.uploaded_weights = []
        self.uploaded_models = []
        tot_samples = 0
        for client in active_clients:
            client_time_cost = client.train_time_cost['total_cost'] / client.train_time_cost['num_rounds'] + \
                    client.send_time_cost['total_cost'] / client.send_time_cost['num_rounds']
            if client_time_cost <= self.time_threthold:
                tot_samples += client.train_samples
                self.uploaded_weights.append(client.train_samples)
                self.uploaded_models.append(client.model.base)
        for i, w in enumerate(self.uploaded_weights):
            self.uploaded_weights[i] = w / tot_samples

    def train_metrics(self):
        if self.eval_new_clients and self.num_new_clients > 0:
            return [0], [1], [0]
        
        num_samples = []
        losses = []
        proto_losses_body = []
        proto_losses_per = []

        for c in self.clients:
            cl, ns, pl_b, pl_p = c.train_metrics()
            num_samples.append(ns)
            losses.append(cl*1.0)
            proto_losses_body.append(pl_b*1.0)
            proto_losses_per.append(pl_p*1.0)

            self.losses_by_clients[c.id].append(cl*1.0 / ns)
            self.losses_b_by_clients[c.id].append(pl_b*1.0 / ns)
            self.losses_p_by_clients[c.id].append(pl_p*1.0 / ns)

        ids = [c.id for c in self.clients]

        return ids, num_samples, losses, proto_losses_body, proto_losses_per

    def evaluate(self, acc=None, loss=None):
        stats = self.test_metrics()
        stats_train = self.train_metrics()

        accs = [a / n for a, n in zip(stats[2], stats[1])]
        self.lta = accs

        if self.test_way == "ind":
            test_acc = np.mean(accs)
            train_loss = np.mean([l / n for l, n in zip(stats_train[2], stats_train[1])])
            proto_loss_body = np.mean([l / n for l, n in zip(stats_train[3], stats_train[1])])
            proto_loss_per = np.mean([l / n for l, n in zip(stats_train[4], stats_train[1])])
        
        else:
            test_acc = sum(stats[2])*1.0 / sum(stats[1])
            train_loss = sum(stats_train[2])*1.0 / sum(stats_train[1])
            proto_loss_body = sum(stats_train[3])*1.0 / sum(stats_train[1])
            proto_loss_per = sum(stats_train[4])*1.0 / sum(stats_train[1])


        if acc == None:
            self.rs_test_acc.append(test_acc)
        else:
            acc.append(test_acc)
        
        if loss == None:
            self.rs_train_loss.append(train_loss)
        else:
            loss.append(train_loss)

        self.rs_proto_loss_body.append(proto_loss_body)
        self.rs_proto_loss_per.append(proto_loss_per)

        print(f"------Metrics: {self.test_way}-------")
        print("Averaged Train Loss: {:.4f}".format(train_loss))
        print("Averaged Train Proto Loss Body: {:.4f}".format(proto_loss_body))
        print("Averaged Train Proto Loss Per: {:.4f}".format(proto_loss_per))
        print("Averaged Test Accurancy: {:.4f}".format(test_acc))
        print("Std Test Accurancy: {:.4f}".format(np.std(accs)))

    def save_results(self):
        data_type = f"./results/{self.data_dir.split('/')[-2]}/{self.data_dir.split('/')[-1]}/"
        if not os.path.exists(data_type):
            os.makedirs(data_type)

        algo = self.algorithm + '_' + self.model_name
        result_path = data_type + algo + "/"
        if not os.path.exists(result_path):
            os.makedirs(result_path)

        if not os.path.exists(result_path + "test/"):
            os.makedirs(result_path + "test/")

        if (len(self.rs_test_acc)):
            algo = self.goal + "_" + str(self.times)
            file_path = result_path + "test/{}.h5".format(algo)
            print("File path: " + file_path)

            with h5py.File(file_path, 'w') as hf:
                hf.create_dataset('rs_test_acc', data=self.rs_test_acc)
                hf.create_dataset('rs_train_loss', data=self.rs_train_loss)
                hf.create_dataset('rs_proto_loss_body', data=self.rs_proto_loss_body)
                hf.create_dataset('rs_proto_loss_per', data=self.rs_proto_loss_per)
                hf.create_dataset('last_acc_by_clients', data=self.lta)

            if not os.path.exists(result_path + "lbc/"):
                os.makedirs(result_path + "lbc/")

            if not os.path.exists(result_path + "lbbc/"):
                os.makedirs(result_path + "lbbc/")

            if not os.path.exists(result_path + "lpbc/"):
                os.makedirs(result_path + "lpbc/")

            if not os.path.exists(result_path + "acc/"):
                os.makedirs(result_path + "acc/")

            with open(result_path + "lbc/{}.pkl".format(algo), 'wb') as f:
                pickle.dump(self.losses_by_clients, f)

            with open(result_path + "lbbc/{}.pkl".format(algo), 'wb') as f:
                pickle.dump(self.losses_b_by_clients, f)

            with open(result_path + "lpbc/{}.pkl".format(algo), 'wb') as f:
                pickle.dump(self.losses_p_by_clients, f)

            with open(result_path + f"acc/{algo}.pkl", 'wb') as f:
                pickle.dump(self.acc_by_clients, f)




                