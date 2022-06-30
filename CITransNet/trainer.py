import os
import sys
import numpy as np
from statistics import mean
import random
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import json
import shutil
from time import time

from base import BaseTrainer
from network import TransformerMechanism
from utilities import misreportOptimization
def loss_function(mechanism,lamb,rho,batch,trueValuations,misreports):
    from utilities import loss
    return loss(mechanism,lamb,rho,batch,trueValuations,misreports)


class Trainer(BaseTrainer):
    def __init__(self, args):
        super(Trainer, self).__init__(args)
        self.set_data(args)
        self.set_model(args)
        self.start_epoch = 0
        self.rho = args.rho
        self.lamb = args.lamb * torch.ones(args.n_bidder).to(args.device)
        self.n_iter = 0

    def set_data(self, args):
        def load_data(dir, size):
            data = [np.load(os.path.join(dir, 'trueValuations.npy')).astype(np.float32),
                    np.load(os.path.join(dir, 'Agent_names_idxs.npy')),
                    np.load(os.path.join(dir, 'Object_names_idxs.npy'))]
            if args.continuous_context:
                for i in [1, 2]:
                    data[i] = data[i].astype(np.float32)
            else:
                for i in [1, 2]:
                    data[i] = data[i].astype(np.int64)
            if size is not None:
                data = [x[:self.train_size] for x in data]
            return tuple(data)

        if not args.test:
            self.train_dir = os.path.join(args.data_dir, args.training_set)
            self.train_size = args.train_size
            self.train_data = load_data(self.train_dir, self.train_size)
            self.train_size = len(self.train_data[0])

            self.misreports = np.random.uniform(args.v_min, args.v_max,
                                                size=(self.train_size, args.n_misreport_init_train,
                                                      args.n_bidder, args.n_item))

        self.test_dir = os.path.join(args.data_dir, args.test_set)
        self.test_size = args.test_size
        self.test_data = load_data(self.test_dir, self.test_size)
        self.test_size = len(self.test_data[0])

    def set_model(self, args):
        self.mechanism = TransformerMechanism(args.n_bidder_type, args.n_item_type, args.d_emb,
                                              args.n_layer, args.n_head, args.d_hidden,
                                              args.v_min, args.v_max, args.continuous_context,
                                              args.cond_prob).to(args.device)
        if args.data_parallel:
            self.mechanism = nn.DataParallel(self.mechanism)
        self.optimizer = torch.optim.Adam(self.mechanism.parameters(), lr=args.learning_rate)

    def save(self, epoch, tag=None):
        save_variable_dict = {
            'mechanism': self.mechanism.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'start_epoch': epoch+1,
            'rho': self.rho,
            'lamb': self.lamb,
            'n_iter': self.n_iter,
            'misreports': self.misreports,
        }
        self.save_model(self.args, save_variable_dict, tag=tag)

    def load(self, tag=None):
        fname = 'checkpoint' if tag is None else f'checkpoint-{tag}'
        try:
            ckpt_path = os.path.join(self.args.save_dir, fname)
        except:
            return
        if os.path.exists(ckpt_path):
            logging.info(f'load checkpoint from {ckpt_path}')
            ckpt = torch.load(ckpt_path)
            self.mechanism.load_state_dict(ckpt['mechanism'])
            self.optimizer.load_state_dict(ckpt['optimizer'])
            self.start_epoch = ckpt['start_epoch']
            self.rho = ckpt['rho']
            self.lamb = ckpt['lamb']
            self.n_iter = ckpt['n_iter']
            if 'misreports' in ckpt:
                self.misreports = ckpt['misreports']

    def train(self, args):
        self.load()
        for epoch in range(self.start_epoch, args.n_epoch):
            profit_sum = 0
            regret_sum = 0
            regret_max = 0
            loss_sum = 0
            for i in tqdm(range(0, self.train_size, args.batch_size)):
                self.n_iter += 1
                batch_indices = np.random.choice(self.train_size, args.batch_size)
                self.misreports = misreportOptimization(self.mechanism, batch_indices, self.train_data, self.misreports,
                                                   args.r_train, args.gamma, args.v_min, args.v_max)
                loss, regret_mean_bidder, regret_max_batch, profit = \
                    loss_function(self.mechanism, self.lamb, self.rho, batch_indices, self.train_data, self.misreports)
                loss_sum += loss.item() * len(batch_indices)
                regret_sum += regret_mean_bidder.mean().item() * len(batch_indices)
                regret_max = max(regret_max, regret_max_batch.item())
                profit_sum += profit.item() * len(batch_indices)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                if self.n_iter % args.lamb_update_freq  == 0:
                    self.lamb += self.rho * regret_mean_bidder.detach()

            if (epoch + 1) % 2 == 0:
                self.rho += args.delta_rho
            logging.info(f"Train: epoch={epoch + 1}, loss={loss_sum/self.train_size:.5}, "
                         f"profit={(profit_sum)/self.train_size:.5}, "
                         f"regret={(regret_sum)/self.train_size:.5}, regret_max={regret_max:.5}")
            logging.info(f"Train: rho={self.rho}, lamb={self.lamb.mean().item()}, ")
            if (epoch + 1) % args.save_freq == 0:
                self.save(epoch)
            if (epoch + 1) % args.eval_freq == 0:
                self.test(args, valid=True)
                self.save(epoch, tag=epoch)

        logging.info('Final test')
        self.save(args.n_epoch-1)
        self.test(args)

    def test(self, args, valid=False, load=False):
        if load:
            self.load(args.test_ckpt_tag)
        if valid == False and args.data_parallel_test:
            self.mechanism = nn.DataParallel(self.mechanism)
        if len(self.lamb) != args.n_bidder:
            self.lamb = args.lamb * torch.ones(args.n_bidder).to(args.device)
        if valid:
            data_size = args.batch_test * 10
            indices = np.random.choice(self.test_size, data_size)
            data = tuple([x[indices] for x in self.test_data])
        else:
            data_size = self.test_size
            data = self.test_data
        misreports = np.random.uniform(args.v_min, args.v_max, size=(data_size, args.n_misreport_init, args.n_bidder, args.n_item))
        indices = np.arange(data_size)
        profit_sum = 0.0
        regret_sum = 0.0
        regret_max = 0.0
        loss_sum = 0.0
        n_iter = 0.0
        for i in tqdm(range(0, data_size, args.batch_test)):
            batch_indices = indices[i:i+args.batch_test]
            n_iter += len(batch_indices)
            misreports = misreportOptimization(self.mechanism, batch_indices, data, misreports,
                                               args.r_test, args.gamma, args.v_min, args.v_max)
            with torch.no_grad():
                loss, regret_mean_bidder, regret_max_batch, profit = \
                    loss_function(self.mechanism, self.lamb, self.rho, batch_indices, data, misreports)

            loss_sum += loss.item() * len(batch_indices)
            regret_sum += regret_mean_bidder.mean().item() * len(batch_indices)
            regret_max = max(regret_max, regret_max_batch.item())
            profit_sum += profit.item() * len(batch_indices)

            if valid == False:
                logging.info(f"profit={(profit_sum)/n_iter:.5}, regret={(regret_sum)/n_iter:.5}")

        logging.info(f"Test: loss={loss_sum/data_size:.5}, profit={(profit_sum)/data_size:.5}, "
                     f"regret={(regret_sum)/data_size:.5}, regret_max={regret_max:.5}")
