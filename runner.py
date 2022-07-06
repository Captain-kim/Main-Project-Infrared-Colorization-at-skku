#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
  @Email:  guangmingwu2010@gmail.com
  @Copyright: go-hiroaki
  @License: MIT
"""
import os
import time
import shutil
import metrics
import losses
import numpy as np
import pandas as pd
import warnings
from tqdm import tqdm
from utils import create_discriminator
import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.autograd as autograd
import torch.nn as nn
import cv2

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

Src_DIR = os.path.dirname(os.path.abspath(__file__))
Logs_DIR = os.path.join(Src_DIR, '../logs')
Checkpoint_DIR = os.path.join(Src_DIR, '../checkpoint')

if not os.path.exists(Logs_DIR):
    os.mkdir(Logs_DIR)
    os.mkdir(os.path.join(Logs_DIR, 'raw'))
    os.mkdir(os.path.join(Logs_DIR, 'curve'))
    os.mkdir(os.path.join(Logs_DIR, 'snapshot'))

if not os.path.exists(Checkpoint_DIR):
    os.mkdir(Checkpoint_DIR)

def get_grid(input, is_real=True):
    if is_real:
        grid = torch.FloatTensor(input.shape).fill_(1.0)

    elif not is_real:
        grid = torch.FloatTensor(input.shape).fill_(0.0)

    return grid

def compute_gradient_penalty(D, input_samples, real_samples, fake_samples):
    alpha = Tensor(np.random.random((real_samples.size(0), 1, 1, 1)))
    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates = D(input_samples, interpolates)
    # For PatchGAN
    fake = Variable(Tensor(real_samples.shape[0], 1, 30, 30).fill_(1.0), requires_grad=False)
    # Get gradient w.r.t. interpolates
    gradients = autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty

class Base(object):
    def __init__(self, args, method, is_multi=False, criterion='L1Loss', metric='PSNR'):
        self.args = args
        self.method = method
        self.is_multi = is_multi
        self.date = time.strftime("%h%d_%H")
        if int(args.alpha) == 0:
            self.method += '-nop'
        self.repr = "{}_{}_{}".format(
            self.method, self.args.trigger, self.args.terminal)
        self.epoch = 0
        self.iter = 0
        self.logs = []
        self.criterion = eval("{}.{}()".format('losses', criterion))
        self.evaluator = eval("{}.{}()".format('metrics', metric))
        self.snapshot = os.path.join(Logs_DIR, "snapshot", self.method)
        if not os.path.exists(self.snapshot):
            os.makedirs(self.snapshot)
        else:
            shutil.rmtree(self.snapshot)
            os.makedirs(self.snapshot)
        
        self.header = ["epoch", "iter"]
        for stage in ['trn', 'val']:
            for key in [repr(self.criterion),repr(self.evaluator),"FPS"]:
                self.header.append("{}_{}".format(stage, key))

    def logging(self, verbose=True):
        self.logs.append([self.epoch, self.iter] +
                         self.trn_log + self.val_log)
        if verbose:
            str_a = ['{}:{:05d}'.format(k,v) for k,v in zip(self.header[:2], [self.epoch, self.iter])]
            str_b = ['{}:{:.2f}'.format(k,v) for k,v in zip(self.header[2:], self.trn_log + self.val_log)]
            print(', '.join(str_a + str_b))

    def save_log(self):
        self.logs = pd.DataFrame(self.logs,
                                 columns=self.header)
        self.logs.to_csv(os.path.join(Logs_DIR, 'raw', '{}.csv'.format(self.repr)), index=False, float_format='%.3f')

        speed_info = [self.repr, self.logs.iloc[:, 4].mean(), self.logs.iloc[:, 7].mean()]
        df = pd.DataFrame([speed_info],
                          columns=["experiment", self.header[4], self.header[7]])
        if os.path.exists(os.path.join(Logs_DIR, 'speed.csv')):
            prev_df = pd.read_csv(os.path.join(Logs_DIR, 'speed.csv'))
            df = prev_df.append(df)
        df.to_csv(os.path.join(Logs_DIR, 'speed.csv'), index=False, float_format='%.3f')

    def save_checkpoint(self, net):
        torch.save(net.state_dict(), os.path.join(Checkpoint_DIR, "{}.pth".format(self.repr)))

    def save_checkpoint_custom(self, net):
        torch.save(net[0].state_dict(), os.path.join(Checkpoint_DIR, "0" + "{}.pth".format(self.repr)))
        torch.save(net[1].state_dict(), os.path.join(Checkpoint_DIR, "1" + "{}.pth".format(self.repr)))
        torch.save(net[2].state_dict(), os.path.join(Checkpoint_DIR, "2" + "{}.pth".format(self.repr)))

    def learning_curve(self, idxs=[2,3,5,6]):
        import seaborn as sns
        import matplotlib.pyplot as plt
        plt.switch_backend('agg')
        # set style
        sns.set_context("paper", font_scale=1.5,)
        # sns.set_style("ticks", {
        #     "font.family": "Times New Roman",
        #     "font.serif": ["Times", "Palatino", "serif"]})

        for idx in idxs:
            plt.plot(self.logs[self.args.trigger],
                     self.logs[self.header[idx]], label=self.header[idx])
        plt.ylabel(" {} / {} ".format(repr(self.criterion), repr(self.evaluator)))
        if self.args.trigger == 'epoch':
            plt.xlabel("Epochs")
        else:
            plt.xlabel("Iterations")
        plt.suptitle("Training log of {}".format(self.method))
        plt.legend(bbox_to_anchor=(1.01, 1), loc=2, borderaxespad=0.)
        plt.savefig(os.path.join(Logs_DIR, 'curve', '{}.png'.format(self.repr)),
                    format='png', bbox_inches='tight', dpi=144)

    def save_snapshot(self, src, tar, gen, dataset):
        """
          Args:
            src: (tensor) tensor of src
            tar: (tensor) tensor of tar
            gen: (tensor) tensor of prediction
        """
        import random
        from skimage.io import imsave

        # transfer to cpu
        idx = 0
        if self.args.cuda:
            src = src.cpu()
            tar = tar.cpu()
            gen = gen.cpu()
        if '3D' in dataset.ver:
            src = src.numpy()[idx, :, idx].transpose((1, 2, 0))
            tar = tar.numpy()[idx, :, idx].transpose((1, 2, 0))
            gen = gen.numpy()[idx, :, idx].transpose((1, 2, 0))
        else:
            src = src.numpy()[idx].transpose((1, 2, 0))
            tar = tar.numpy()[idx].transpose((1, 2, 0))
            gen = gen.numpy()[idx].transpose((1, 2, 0))
        src_type, tar_type = dataset.ver.split("2")
        if "LAB" in src_type:
            src_img = dataset._lab2img(src)
        else:
            src_img = dataset._rgb2img(src)
        if "LAB" != tar_type:
            tar_img = dataset._rgb2img(tar)
            gen_img = dataset._rgb2img(gen)
        else:
            tar_img = dataset._lab2img(tar)
            gen_img = dataset._lab2img(gen)
        vis_img = np.concatenate([src_img, tar_img, gen_img], axis=0)
        # save image
        imsave(os.path.join(self.snapshot, '{}_iter-{:05d}.png'.format(self.method, self.iter)), vis_img)


class Trainer(Base):
    def training(self, net, datasets):
        """
          Args:
            net: (object) net & optimizer
            datasets : (list) [train, val] dataset object
        """
        args = self.args
        best_trn_perform, best_val_perform = -1, -1
        steps = len(datasets[0]) // args.batch_size
        if steps * args.batch_size < len(datasets[0]):
            steps += 1

        if args.trigger == 'epoch':
            args.epochs = args.terminal
            args.iters = steps * args.terminal
            args.iter_interval = steps * args.interval
        else:
            args.epochs = args.terminal // steps + 1
            args.iters = args.terminal
            args.iter_interval = args.interval

        net.train()
        trn_loss, trn_acc = [], []
        start = time.time()
        for epoch in range(1, args.epochs + 1):
            self.epoch = epoch
            # setup data loader
            data_loader = DataLoader(datasets[0], args.batch_size, num_workers=4,
                                     shuffle=True, pin_memory=True,)
            for idx, sample in enumerate(data_loader):
                self.iter += 1
                if self.iter > args.iters:
                    self.iter -= 1
                    break
                # get tensors from sample
                x = sample["src"]
                y = sample["tar"]
                if args.cuda:
                    x = x.cuda()
                    y = y.cuda()
                # forwading
                gen_y = net(x)
                loss = self.criterion(gen_y, y)
                # update parameters
                net.optimizer.zero_grad()
                loss.backward()
                net.optimizer.step()
                # update taining condition
                trn_loss.append(loss.detach().item())
                trn_acc.append(self.evaluator(gen_y.detach(), y.detach()).item())
                # validating
                if self.iter % args.iter_interval == 0:
                    trn_fps = (args.iter_interval * args.batch_size) / (time.time() - start)
                    self.trn_log = [round(sum(trn_loss) / len(trn_loss), 3), 
                                    round(sum(trn_acc) / len(trn_acc), 3),
                                    round(trn_fps, 3)]
 
                    self.validating(net, datasets[1])
                    self.logging(verbose=True)
                    if self.val_log[1] >= best_val_perform:
                        best_trn_perform = self.trn_log[1]
                        best_val_perform = self.val_log[1]
                        checkpoint_info = [self.repr, self.epoch, self.iter,
                                           best_trn_perform, best_val_perform]
                        # save better checkpoint
                        self.save_checkpoint(net)
                    # reinitialize
                    start = time.time()
                    trn_loss, trn_acc = [], []
                    net.train()

        df = pd.DataFrame([checkpoint_info],
                          columns=["experiment", "best_epoch", "best_iter", self.header[3], self.header[6]])
        if os.path.exists(os.path.join(Checkpoint_DIR, 'checkpoint.csv')):
            prev_df = pd.read_csv(os.path.join(Checkpoint_DIR, 'checkpoint.csv'))
            df = prev_df.append(df)
        df.to_csv(os.path.join(Checkpoint_DIR, 'checkpoint.csv'), index=False, float_format='%.3f')

        print("Best {} Performance: \n".format(repr(self.evaluator)))
        print("\t Trn:", best_trn_perform)
        print("\t Val:", best_val_perform)

    def validating(self, net, dataset):
        """
          Args:
            net: (object) pytorch net
            batch_size: (int)
            dataset : (object) dataset
          return [loss, acc]
        """
        args = self.args
        data_loader = DataLoader(dataset, args.batch_size, num_workers=4,
                                 shuffle=True, pin_memory=True,)
        val_loss, val_acc = [], []
        start = time.time()
        net.eval()
        with torch.set_grad_enabled(False):
            for idx, sample in enumerate(data_loader):
                # get tensors from sample
                x = sample["src"]
                y = sample["tar"]
                if args.cuda:
                    x = x.cuda()
                    y = y.cuda()
               # forwading
                gen_y = net(x)
                val_loss.append(self.criterion(gen_y.detach(), y.detach()).item())
                val_acc.append(self.evaluator(gen_y.detach(), y.detach()).item())

        val_fps = (len(val_loss) * args.batch_size) / (time.time() - start)
        self.val_log = [round(sum(val_loss) / len(val_loss), 3), 
                        round(sum(val_acc) / len(val_acc), 3),
                        round(val_fps, 3)]
        self.save_snapshot(x.detach(), y.detach(), gen_y.detach(), dataset)


class MY_Trainer(Trainer):
    def training(self, net, datasets):
        """
          Args:
            net: (object) net & optimizer
            datasets : (list) [train, val] dataset object
        """
        args = self.args
        best_trn_perform, best_val_perform = -1, -1
        steps = len(datasets[0]) // args.batch_size
        if steps * args.batch_size < len(datasets[0]):
            steps += 1

        if args.trigger == 'epoch':
            args.epochs = args.terminal
            args.iters = steps * args.terminal
            args.iter_interval = steps * args.interval
        else:
            args.epochs = args.terminal // steps + 1
            args.iters = args.terminal
            args.iter_interval = args.interval

        net.train()
        trn_loss, trn_acc = [], []
        start = time.time()
        criterionR = losses.L1Loss()
        criterionG = losses.L1Loss()
        criterionB = losses.L1Loss()
        # criterionA = losses.DSSIMLoss()
        # criterionB = losses.L1Loss()
        # criterionExtra = losses.VGG16Loss()

        for epoch in range(1, args.epochs + 1):
            self.epoch = epoch
            # setup data loader
            data_loader = DataLoader(datasets[0], args.batch_size, num_workers=4,
                                     shuffle=True, pin_memory=True, )
            for idx, sample in enumerate(tqdm(data_loader)):
                self.iter += 1
                if self.iter > args.iters:
                    self.iter -= 1
                    break
                # get tensors from sample
                x = sample["src"]
                y = sample["tar"]
                if args.cuda:
                    x = x.cuda()
                    y = y.cuda()
                # forwading
                # gen_l, gen_ab = net(x)
                gen_R, gen_G, gen_B = net(x)
                # gen_y = torch.cat([gen_l, gen_ab], dim=1)
                gen_y = torch.cat([gen_R, gen_G, gen_B], dim=1)
                # loss_l = self.criterion(gen_y[:,:1,:,:], y[:,:1,:,:])
                # loss_ab = self.criterion(gen_y[:,1:,:,:], y[:,1:,:,:])
                loss_R = criterionR(gen_y[:, 0, :, :], y[:, 0, :, :])
                loss_G = criterionG(gen_y[:, 1, :, :], y[:, 1, :, :])
                loss_B = criterionB(gen_y[:, 2, :, :], y[:, 2, :, :])

                # loss_l = criterionA(gen_y[:, :1, :, :], y[:, :1, :, :])
                # loss_ab = criterionB(gen_y[:, 1:, :, :], y[:, 1:, :, :])
                # loss_VGG = criterionExtra(gen_y, y)
                # loss = loss_l + loss_ab + args.alpha * loss_lab
                loss = loss_R + loss_G + loss_B
                # update parameters
                net.optimizer.zero_grad()
                loss.backward()
                net.optimizer.step()
                # update taining condition
                trn_loss.append(loss.detach().item())
                trn_acc.append(self.evaluator(gen_y.detach(), y.detach()).item())
                # validating
                if self.iter % args.iter_interval == 0:
                    trn_fps = (args.iter_interval * args.batch_size) / (time.time() - start)
                    self.trn_log = [round(sum(trn_loss) / len(trn_loss), 3),
                                    round(sum(trn_acc) / len(trn_acc), 3),
                                    round(trn_fps, 3)]

                    self.validating(net, datasets[1])
                    self.logging(verbose=True)
                    if self.val_log[1] >= best_val_perform:
                        best_trn_perform = self.trn_log[1]
                        best_val_perform = self.val_log[1]
                        checkpoint_info = [self.repr, self.epoch, self.iter,
                                           best_trn_perform, best_val_perform]
                        # save better checkpoint
                        self.save_checkpoint(net)
                    # reinitialize
                    start = time.time()
                    trn_loss, trn_acc = [], []
                    net.train()

        df = pd.DataFrame([checkpoint_info],
                          columns=["experiment", "best_epoch", "best_iter", self.header[3], self.header[6]])
        if os.path.exists(os.path.join(Checkpoint_DIR, 'checkpoint.csv')):
            prev_df = pd.read_csv(os.path.join(Checkpoint_DIR, 'checkpoint.csv'))
            df = prev_df.append(df)
        df.to_csv(os.path.join(Checkpoint_DIR, 'checkpoint.csv'), index=False, float_format='%.3f')

        print("Best {} Performance: \n".format(repr(self.evaluator)))
        print("\t Trn:", best_trn_perform)
        print("\t Val:", best_val_perform)

    def validating(self, net, dataset):
        """
          Args:
            net: (object) pytorch net
            batch_size: (int)
            dataset : (object) dataset
          return [loss, acc]
        """
        args = self.args
        data_loader = DataLoader(dataset, args.batch_size, num_workers=4,
                                 shuffle=False, pin_memory=True, )
        val_loss, val_acc = [], []
        start = time.time()
        net.eval()
        with torch.set_grad_enabled(False):
            for idx, sample in enumerate(data_loader):
                # get tensors from sample
                x = sample["src"]
                y = sample["tar"]
                if args.cuda:
                    x = x.cuda()
                    y = y.cuda()
                # forwading
                gen_R, gen_G, gen_B = net(x)
                # gen_l, gen_ab = net(x)
                # gen_y = torch.cat([gen_l, gen_ab], dim=1)
                gen_y = torch.cat([gen_R, gen_G, gen_B], dim=1)
                val_loss.append(self.criterion(gen_y.detach(), y.detach()).item())
                val_acc.append(self.evaluator(gen_y.detach(), y.detach()).item())

        val_fps = (len(val_loss) * args.batch_size) / (time.time() - start)
        self.val_log = [round(sum(val_loss) / len(val_loss), 3),
                        round(sum(val_acc) / len(val_acc), 3),
                        round(val_fps, 3)]
        self.save_snapshot(x.detach(), y.detach(), gen_y.detach(), dataset)

class Lab_Trainer(Trainer):
    def training(self, net, datasets):
        """
          Args:
            net: (object) net & optimizer
            datasets : (list) [train, val] dataset object
        """
        args = self.args
        best_trn_perform, best_val_perform = -1, -1
        steps = len(datasets[0]) // args.batch_size
        if steps * args.batch_size < len(datasets[0]):
            steps += 1

        if args.trigger == 'epoch':
            args.epochs = args.terminal
            args.iters = steps * args.terminal
            args.iter_interval = steps * args.interval
        else:
            args.epochs = args.terminal // steps + 1
            args.iters = args.terminal
            args.iter_interval = args.interval

        net.train()
        trn_loss, trn_acc = [], []
        start = time.time()
        criterion_L = losses.L1Loss()
        criterion_a = losses.L1Loss()
        criterion_b = losses.L1Loss()
        # criterionA = losses.DSSIMLoss()
        # criterionB = losses.L1Loss()
        # criterionExtra = losses.VGG16Loss()

        for epoch in range(1, args.epochs + 1):
            self.epoch = epoch
            # setup data loader
            data_loader = DataLoader(datasets[0], args.batch_size, num_workers=4,
                                     shuffle=True, pin_memory=True, )
            for idx, sample in enumerate(tqdm(data_loader)):
                self.iter += 1
                if self.iter > args.iters:
                    self.iter -= 1
                    break
                # get tensors from sample
                x = sample["src"]
                y = sample["tar"]
                if args.cuda:
                    x = x.cuda()
                    y = y.cuda()
                # forwading
                # gen_l, gen_ab = net(x)
                gen_L, gen_a, gen_b = net(x)
                # gen_y = torch.cat([gen_l, gen_ab], dim=1)
                gen_y = torch.cat([gen_L, gen_a, gen_b], dim=1)
                # loss_l = self.criterion(gen_y[:,:1,:,:], y[:,:1,:,:])
                # loss_ab = self.criterion(gen_y[:,1:,:,:], y[:,1:,:,:])
                loss_L = criterion_L(gen_y[:, 0, :, :], y[:, 0, :, :])
                loss_a = criterion_a(gen_y[:, 1, :, :], y[:, 1, :, :])
                loss_b = criterion_b(gen_y[:, 2, :, :], y[:, 2, :, :])

                # loss_l = criterionA(gen_y[:, :1, :, :], y[:, :1, :, :])
                # loss_ab = criterionB(gen_y[:, 1:, :, :], y[:, 1:, :, :])
                # loss_VGG = criterionExtra(gen_y, y)
                # loss = loss_l + loss_ab + args.alpha * loss_lab
                loss = loss_L + loss_a + loss_b
                # update parameters
                net.optimizer.zero_grad()
                loss.backward()
                net.optimizer.step()
                # update taining condition
                trn_loss.append(loss.detach().item())
                trn_acc.append(self.evaluator(gen_y.detach(), y.detach()).item())
                # validating
                if self.iter % args.iter_interval == 0:
                    trn_fps = (args.iter_interval * args.batch_size) / (time.time() - start)
                    self.trn_log = [round(sum(trn_loss) / len(trn_loss), 3),
                                    round(sum(trn_acc) / len(trn_acc), 3),
                                    round(trn_fps, 3)]

                    self.validating(net, datasets[1])
                    self.logging(verbose=True)
                    if self.val_log[1] >= best_val_perform:
                        best_trn_perform = self.trn_log[1]
                        best_val_perform = self.val_log[1]
                        checkpoint_info = [self.repr, self.epoch, self.iter,
                                           best_trn_perform, best_val_perform]
                        # save better checkpoint
                        self.save_checkpoint(net)
                    # reinitialize
                    start = time.time()
                    trn_loss, trn_acc = [], []
                    net.train()

        df = pd.DataFrame([checkpoint_info],
                          columns=["experiment", "best_epoch", "best_iter", self.header[3], self.header[6]])
        if os.path.exists(os.path.join(Checkpoint_DIR, 'checkpoint.csv')):
            prev_df = pd.read_csv(os.path.join(Checkpoint_DIR, 'checkpoint.csv'))
            df = prev_df.append(df)
        df.to_csv(os.path.join(Checkpoint_DIR, 'checkpoint.csv'), index=False, float_format='%.3f')

        print("Best {} Performance: \n".format(repr(self.evaluator)))
        print("\t Trn:", best_trn_perform)
        print("\t Val:", best_val_perform)

    def validating(self, net, dataset):
        """
          Args:
            net: (object) pytorch net
            batch_size: (int)
            dataset : (object) dataset
          return [loss, acc]
        """
        args = self.args
        data_loader = DataLoader(dataset, args.batch_size, num_workers=4,
                                 shuffle=False, pin_memory=True, )
        val_loss, val_acc = [], []
        start = time.time()
        net.eval()
        with torch.set_grad_enabled(False):
            for idx, sample in enumerate(data_loader):
                # get tensors from sample
                x = sample["src"]
                y = sample["tar"]
                if args.cuda:
                    x = x.cuda()
                    y = y.cuda()
                # forwading
                gen_L, gen_a, gen_b = net(x)
                # gen_l, gen_ab = net(x)
                # gen_y = torch.cat([gen_l, gen_ab], dim=1)
                gen_y = torch.cat([gen_L, gen_a, gen_b], dim=1)
                val_loss.append(self.criterion(gen_y.detach(), y.detach()).item())
                val_acc.append(self.evaluator(gen_y.detach(), y.detach()).item())

        val_fps = (len(val_loss) * args.batch_size) / (time.time() - start)
        self.val_log = [round(sum(val_loss) / len(val_loss), 3),
                        round(sum(val_acc) / len(val_acc), 3),
                        round(val_fps, 3)]
        self.save_snapshot(x.detach(), y.detach(), gen_y.detach(), dataset)

class VAE_LAB_Trainer(Trainer):
    def training(self, net, datasets):
        """
          Args:
            net: (object) net & optimizer
            datasets : (list) [train, val] dataset object
        """
        args = self.args
        best_trn_perform, best_val_perform = -1, -1
        steps = len(datasets[0]) // args.batch_size
        if steps * args.batch_size < len(datasets[0]):
            steps += 1

        if args.trigger == 'epoch':
            args.epochs = args.terminal
            args.iters = steps * args.terminal
            args.iter_interval = steps * args.interval
        else:
            args.epochs = args.terminal // steps + 1
            args.iters = args.terminal
            args.iter_interval = args.interval

        net.train()
        trn_loss, trn_acc = [], []
        start = time.time()
        criterion_L = losses.L1Loss()
        criterion_a = losses.L1Loss()
        criterion_b = losses.L1Loss()
        criterion_VAE = losses.VAELoss()
        # criterionA = losses.DSSIMLoss()
        # criterionB = losses.L1Loss()
        criterionExtra = losses.VGG16Loss()

        for epoch in range(1, args.epochs + 1):
            self.epoch = epoch
            # setup data loader
            data_loader = DataLoader(datasets[0], args.batch_size, num_workers=4,
                                     shuffle=True, pin_memory=True, )
            for idx, sample in enumerate(tqdm(data_loader)):
                self.iter += 1
                if self.iter > args.iters:
                    self.iter -= 1
                    break
                # get tensors from sample
                x = sample["src"]
                y = sample["tar"]
                if args.cuda:
                    x = x.cuda()
                    y = y.cuda()
                # forwading
                # gen_l, gen_ab = net(x)
                gen_L, gen_a, gen_b, mu, logvar = net(x)
                # gen_y = torch.cat([gen_l, gen_ab], dim=1)
                gen_y = torch.cat([gen_L, gen_a, gen_b], dim=1)
                # loss_l = self.criterion(gen_y[:,:1,:,:], y[:,:1,:,:])
                # loss_ab = self.criterion(gen_y[:,1:,:,:], y[:,1:,:,:])
                loss_L = criterion_L(gen_y[:, 0, :, :], y[:, 0, :, :])
                loss_a = criterion_a(gen_y[:, 1, :, :], y[:, 1, :, :])
                loss_b = criterion_b(gen_y[:, 2, :, :], y[:, 2, :, :])
                loss_VAE = criterion_VAE(gen_y, x, mu, logvar)
                loss_VGG = criterionExtra(gen_y, y)
                # loss_l = criterionA(gen_y[:, :1, :, :], y[:, :1, :, :])
                # loss_ab = criterionB(gen_y[:, 1:, :, :], y[:, 1:, :, :])
                # loss_VGG = criterionExtra(gen_y, y)
                # loss = loss_l + loss_ab + args.alpha * loss_lab
                loss = loss_L + loss_a + loss_b + loss_VAE + args.alpha * loss_VGG
                # update parameters
                net.optimizer.zero_grad()
                loss.backward()
                net.optimizer.step()
                # update taining condition
                trn_loss.append(loss.detach().item())
                trn_acc.append(self.evaluator(gen_y.detach(), y.detach()).item())
                # validating
                if self.iter % args.iter_interval == 0:
                    trn_fps = (args.iter_interval * args.batch_size) / (time.time() - start)
                    self.trn_log = [round(sum(trn_loss) / len(trn_loss), 3),
                                    round(sum(trn_acc) / len(trn_acc), 3),
                                    round(trn_fps, 3)]

                    self.validating(net, datasets[1])
                    self.logging(verbose=True)
                    if self.val_log[1] >= best_val_perform:
                        best_trn_perform = self.trn_log[1]
                        best_val_perform = self.val_log[1]
                        checkpoint_info = [self.repr, self.epoch, self.iter,
                                           best_trn_perform, best_val_perform]
                        # save better checkpoint
                        self.save_checkpoint(net)
                    # reinitialize
                    start = time.time()
                    trn_loss, trn_acc = [], []
                    net.train()

        df = pd.DataFrame([checkpoint_info],
                          columns=["experiment", "best_epoch", "best_iter", self.header[3], self.header[6]])
        if os.path.exists(os.path.join(Checkpoint_DIR, 'checkpoint.csv')):
            prev_df = pd.read_csv(os.path.join(Checkpoint_DIR, 'checkpoint.csv'))
            df = prev_df.append(df)
        df.to_csv(os.path.join(Checkpoint_DIR, 'checkpoint.csv'), index=False, float_format='%.3f')

        print("Best {} Performance: \n".format(repr(self.evaluator)))
        print("\t Trn:", best_trn_perform)
        print("\t Val:", best_val_perform)

    def validating(self, net, dataset):
        """
          Args:
            net: (object) pytorch net
            batch_size: (int)
            dataset : (object) dataset
          return [loss, acc]
        """
        args = self.args
        data_loader = DataLoader(dataset, args.batch_size, num_workers=4,
                                 shuffle=False, pin_memory=True, )
        val_loss, val_acc = [], []
        start = time.time()
        net.eval()
        with torch.set_grad_enabled(False):
            for idx, sample in enumerate(data_loader):
                # get tensors from sample
                x = sample["src"]
                y = sample["tar"]
                if args.cuda:
                    x = x.cuda()
                    y = y.cuda()
                # forwading
                gen_L, gen_a, gen_b, mu, logvar = net(x)
                # gen_l, gen_ab = net(x)
                # gen_y = torch.cat([gen_l, gen_ab], dim=1)
                gen_y = torch.cat([gen_L, gen_a, gen_b], dim=1)
                val_loss.append(self.criterion(gen_y.detach(), y.detach()).item())
                val_acc.append(self.evaluator(gen_y.detach(), y.detach()).item())

        val_fps = (len(val_loss) * args.batch_size) / (time.time() - start)
        self.val_log = [round(sum(val_loss) / len(val_loss), 3),
                        round(sum(val_acc) / len(val_acc), 3),
                        round(val_fps, 3)]
        self.save_snapshot(x.detach(), y.detach(), gen_y.detach(), dataset)

class VQ_Trainer(Trainer):
    def training(self, net, datasets):
        """
          Args:
            net: (object) net & optimizer
            datasets : (list) [train, val] dataset object
        """
        args = self.args
        best_trn_perform, best_val_perform = -1, -1
        steps = len(datasets[0]) // args.batch_size
        if steps * args.batch_size < len(datasets[0]):
            steps += 1

        if args.trigger == 'epoch':
            args.epochs = args.terminal
            args.iters = steps * args.terminal
            args.iter_interval = steps * args.interval
        else:
            args.epochs = args.terminal // steps + 1
            args.iters = args.terminal
            args.iter_interval = args.interval

        net.train()
        trn_loss, trn_acc = [], []
        start = time.time()

        criterionA = losses.DSSIMLoss()
        criterionB = losses.L1Loss()
        criterionExtra = losses.VGG16Loss()
        latent_loss_weight = 0.25

        for epoch in range(1, args.epochs + 1):
            self.epoch = epoch
            # setup data loader
            data_loader = DataLoader(datasets[0], args.batch_size, num_workers=4,
                                     shuffle=True, pin_memory=True, )
            for idx, sample in enumerate(tqdm(data_loader)):
                self.iter += 1
                if self.iter > args.iters:
                    self.iter -= 1
                    break
                # get tensors from sample
                x = sample["src"]
                y = sample["tar"]
                if args.cuda:
                    x = x.cuda()
                    y = y.cuda()
                # forwading
                gen_l, gen_a, gen_b, dec, latent_loss = net(x)
                # gen_L, gen_a, gen_b, mu, logvar = net(x)
                # gen_y = torch.cat([gen_l, gen_ab], dim=1)
                gen_y = torch.cat([gen_l, gen_a, gen_b], dim=1)
                latent_loss = latent_loss.mean()
                # loss_l = self.criterion(gen_y[:,:1,:,:], y[:,:1,:,:])
                # loss_ab = self.criterion(gen_y[:,1:,:,:], y[:,1:,:,:])
                # loss_L = criterion_L(gen_y[:, 0, :, :], y[:, 0, :, :])

                loss_l = criterionA(gen_y[:, :1, :, :], y[:, :1, :, :])
                #loss_a = criterionB(gen_y[:, 1, :, :], y[:, 1, :, :])
                #loss_b = criterionB(gen_y[:, 2, :, :], y[:, 2, :, :])
                #loss_l = criterionA(gen_y[:, :1, :, :], y[:, :1, :, :])

                loss_a = criterionB(gen_y[:, 1, :, :], y[:, 1, :, :])
                loss_b = criterionB(gen_y[:, 2, :, :], y[:, 2, :, :])


                loss_VGG = criterionExtra(gen_y, y)
                loss = loss_l + loss_a + loss_b + latent_loss_weight * latent_loss + args.alpha * loss_VGG
                # loss = loss_L + loss_a + loss_b + loss_VAE + args.alpha * loss_VGG
                # update parameters
                net.optimizer.zero_grad()
                loss.backward()
                net.optimizer.step()
                # update taining condition
                trn_loss.append(loss.detach().item())
                trn_acc.append(self.evaluator(gen_y.detach(), y.detach()).item())
                # validating
                if self.iter % args.iter_interval == 0:
                    trn_fps = (args.iter_interval * args.batch_size) / (time.time() - start)
                    self.trn_log = [round(sum(trn_loss) / len(trn_loss), 3),
                                    round(sum(trn_acc) / len(trn_acc), 3),
                                    round(trn_fps, 3)]

                    self.validating(net, datasets[1])
                    self.logging(verbose=True)
                    if self.val_log[1] >= best_val_perform:
                        best_trn_perform = self.trn_log[1]
                        best_val_perform = self.val_log[1]
                        checkpoint_info = [self.repr, self.epoch, self.iter,
                                           best_trn_perform, best_val_perform]
                        # save better checkpoint
                        self.save_checkpoint(net)
                    # reinitialize
                    start = time.time()
                    trn_loss, trn_acc = [], []
                    net.train()

        df = pd.DataFrame([checkpoint_info],
                          columns=["experiment", "best_epoch", "best_iter", self.header[3], self.header[6]])
        if os.path.exists(os.path.join(Checkpoint_DIR, 'checkpoint.csv')):
            prev_df = pd.read_csv(os.path.join(Checkpoint_DIR, 'checkpoint.csv'))
            df = prev_df.append(df)
        df.to_csv(os.path.join(Checkpoint_DIR, 'checkpoint.csv'), index=False, float_format='%.3f')

        print("Best {} Performance: \n".format(repr(self.evaluator)))
        print("\t Trn:", best_trn_perform)
        print("\t Val:", best_val_perform)

    def validating(self, net, dataset):
        """
          Args:
            net: (object) pytorch net
            batch_size: (int)
            dataset : (object) dataset
          return [loss, acc]
        """
        args = self.args
        data_loader = DataLoader(dataset, args.batch_size, num_workers=4,
                                 shuffle=False, pin_memory=True, )
        val_loss, val_acc = [], []
        start = time.time()
        net.eval()
        with torch.set_grad_enabled(False):
            for idx, sample in enumerate(data_loader):
                # get tensors from sample
                x = sample["src"]
                y = sample["tar"]
                if args.cuda:
                    x = x.cuda()
                    y = y.cuda()
                # forwading
                gen_l, gen_a, gen_b, dec, _ = net(x)
                # gen_l, gen_ab = net(x)
                gen_y = torch.cat([gen_l, gen_a, gen_b], dim=1)
                # gen_y = torch.cat([gen_L, gen_a, gen_b], dim=1)

                val_loss.append(self.criterion(gen_y.detach(), y.detach()).item())
                val_acc.append(self.evaluator(gen_y.detach(), y.detach()).item())

        val_fps = (len(val_loss) * args.batch_size) / (time.time() - start)
        self.val_log = [round(sum(val_loss) / len(val_loss), 3),
                        round(sum(val_acc) / len(val_acc), 3),
                        round(val_fps, 3)]
        self.save_snapshot(x.detach(), y.detach(), gen_y.detach(), dataset)

class WGAN_GP_Trainer(Trainer):
    def training(self, net, datasets):
        """
          Args:
            net: (object) net & optimizer
            datasets : (list) [train, val] dataset object
        """
        args = self.args
        best_trn_perform, best_val_perform = -1, -1
        steps = len(datasets[0]) // args.batch_size
        if steps * args.batch_size < len(datasets[0]):
            steps += 1

        if args.trigger == 'epoch':
            args.epochs = args.terminal
            args.iters = steps * args.terminal
            args.iter_interval = steps * args.interval
        else:
            args.epochs = args.terminal // steps + 1
            args.iters = args.terminal
            args.iter_interval = args.interval

        net.train()
        discriminator = create_discriminator()
        discriminator = discriminator.cuda()
        optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=1e-4, betas=(0.5, 0.999))

        trn_loss, trn_acc = [], []
        start = time.time()

        criterionA = losses.DSSIMLoss()
        criterionB = losses.L1Loss()
        criterionExtra = losses.VGG16Loss()
        latent_loss_weight = 0.25

        for epoch in range(1, args.epochs + 1):
            self.epoch = epoch
            # setup data loader
            data_loader = DataLoader(datasets[0], args.batch_size, num_workers=4,
                                     shuffle=True, pin_memory=True, )
            for idx, sample in enumerate(tqdm(data_loader)):
                self.iter += 1
                if self.iter > args.iters:
                    self.iter -= 1
                    break
                # get tensors from sample
                x = sample["src"]
                y = sample["tar"]
                if args.cuda:
                    x = x.cuda()
                    y = y.cuda()
                # forwading
                gen_l, gen_a, gen_b, dec, latent_loss = net(x)
                # gen_L, gen_a, gen_b, mu, logvar = net(x)
                # gen_y = torch.cat([gen_l, gen_ab], dim=1)
                gen_y = torch.cat([gen_l, gen_a, gen_b], dim=1)

                optimizer_D.zero_grad()
                # Fake colorizations
                fake_scalar_d = discriminator(gen_y.detach())

                # True colorizations
                true_scalar_d = discriminator(y)
                gradient_penalty = compute_gradient_penalty(discriminator, y, gen_y)
                loss_D = - torch.mean(true_scalar_d) + torch.mean(fake_scalar_d) + 10 * gradient_penalty

                loss_D.backward(retain_graph=True)
                optimizer_D.step()
                latent_loss = latent_loss.mean()
                # loss_l = self.criterion(gen_y[:,:1,:,:], y[:,:1,:,:])
                # loss_ab = self.criterion(gen_y[:,1:,:,:], y[:,1:,:,:])
                # loss_L = criterion_L(gen_y[:, 0, :, :], y[:, 0, :, :])

                loss_l = criterionA(gen_y[:, :1, :, :], y[:, :1, :, :])
                #loss_a = criterionB(gen_y[:, 1, :, :], y[:, 1, :, :])
                #loss_b = criterionB(gen_y[:, 2, :, :], y[:, 2, :, :])
                #loss_l = criterionA(gen_y[:, :1, :, :], y[:, :1, :, :])

                loss_a = criterionB(gen_y[:, 1, :, :], y[:, 1, :, :])
                loss_b = criterionB(gen_y[:, 2, :, :], y[:, 2, :, :])


                loss_VGG = criterionExtra(gen_y, y)
                # GAN Loss
                fake_scalar = discriminator(gen_y)
                loss_GAN = - torch.mean(fake_scalar)
                loss = loss_l + loss_a + loss_b + latent_loss_weight * latent_loss + 0.05 * loss_GAN + args.alpha * loss_VGG
                # loss = loss_L + loss_a + loss_b + loss_VAE + args.alpha * loss_VGG
                # update parameters
                net.optimizer.zero_grad()
                loss.backward()
                net.optimizer.step()
                # update taining condition
                trn_loss.append(loss.detach().item())
                trn_acc.append(self.evaluator(gen_y.detach(), y.detach()).item())
                # validating
                if self.iter % args.iter_interval == 0:
                    trn_fps = (args.iter_interval * args.batch_size) / (time.time() - start)
                    self.trn_log = [round(sum(trn_loss) / len(trn_loss), 3),
                                    round(sum(trn_acc) / len(trn_acc), 3),
                                    round(trn_fps, 3)]

                    self.validating(net, datasets[1])
                    self.logging(verbose=True)
                    if self.val_log[1] >= best_val_perform:
                        best_trn_perform = self.trn_log[1]
                        best_val_perform = self.val_log[1]
                        checkpoint_info = [self.repr, self.epoch, self.iter,
                                           best_trn_perform, best_val_perform]
                        # save better checkpoint
                        self.save_checkpoint(net)
                    # reinitialize
                    start = time.time()
                    trn_loss, trn_acc = [], []
                    net.train()

        df = pd.DataFrame([checkpoint_info],
                          columns=["experiment", "best_epoch", "best_iter", self.header[3], self.header[6]])
        if os.path.exists(os.path.join(Checkpoint_DIR, 'checkpoint.csv')):
            prev_df = pd.read_csv(os.path.join(Checkpoint_DIR, 'checkpoint.csv'))
            df = prev_df.append(df)
        df.to_csv(os.path.join(Checkpoint_DIR, 'checkpoint.csv'), index=False, float_format='%.3f')

        print("Best {} Performance: \n".format(repr(self.evaluator)))
        print("\t Trn:", best_trn_perform)
        print("\t Val:", best_val_perform)

    def validating(self, net, dataset):
        """
          Args:
            net: (object) pytorch net
            batch_size: (int)
            dataset : (object) dataset
          return [loss, acc]
        """
        args = self.args
        data_loader = DataLoader(dataset, args.batch_size, num_workers=4,
                                 shuffle=False, pin_memory=True, )
        val_loss, val_acc = [], []
        start = time.time()
        net.eval()
        with torch.set_grad_enabled(False):
            for idx, sample in enumerate(data_loader):
                # get tensors from sample
                x = sample["src"]
                y = sample["tar"]
                if args.cuda:
                    x = x.cuda()
                    y = y.cuda()
                # forwading
                gen_l, gen_a, gen_b, _, _ = net(x)
                # gen_l, gen_ab = net(x)
                gen_y = torch.cat([gen_l, gen_a, gen_b], dim=1)
                # gen_y = torch.cat([gen_L, gen_a, gen_b], dim=1)
                val_loss.append(self.criterion(gen_y.detach(), y.detach()).item())
                val_acc.append(self.evaluator(gen_y.detach(), y.detach()).item())

        val_fps = (len(val_loss) * args.batch_size) / (time.time() - start)
        self.val_log = [round(sum(val_loss) / len(val_loss), 3),
                        round(sum(val_acc) / len(val_acc), 3),
                        round(val_fps, 3)]
        self.save_snapshot(x.detach(), y.detach(), gen_y.detach(), dataset)

Tensor = torch.cuda.FloatTensor

def compute_gradient_penalty(D, real_samples, fake_samples):
# Random weight term for interpolation between real and fake samples
    alpha = Tensor(np.random.random((real_samples.size(0), 1, 1, 1)))
    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates = D(interpolates)
    # For PatchGAN
    fake = Variable(Tensor(real_samples.shape[0], 1, 30, 30).fill_(1.0), requires_grad=False)
    # Get gradient w.r.t. interpolates
    gradients = autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty

class PatchDiscriminator(nn.Module):
    def __init__(self):
        super(PatchDiscriminator, self).__init__()

        act = nn.LeakyReLU(0.2, inplace=True)
        input_channel = 3 + 3
        n_df = 64
        norm = nn.InstanceNorm2d

        # 70 x 70
        blocks = []
        blocks += [[nn.Conv2d(input_channel, n_df, kernel_size=4, padding=1, stride=2), act]]
        blocks += [[nn.Conv2d(n_df, 2 * n_df, kernel_size=4, padding=1, stride=2), norm(2 * n_df), act]]
        blocks += [[nn.Conv2d(2 * n_df, 4 * n_df, kernel_size=4, padding=1, stride=2), norm(4 * n_df), act]]
        blocks += [[nn.Conv2d(4 * n_df, 8 * n_df, kernel_size=4, padding=1, stride=1), norm(8 * n_df), act]]
        blocks += [[nn.Conv2d(8 * n_df, 1, kernel_size=4, padding=1, stride=1)]]

        self.n_blocks = len(blocks)
        for i in range(self.n_blocks):
            setattr(self, 'block_{}'.format(i), nn.Sequential(*blocks[i]))

    def forward(self, x):
        result = [x]
        for i in range(self.n_blocks):
            block = getattr(self, 'block_{}'.format(i))
            result.append(block(result[-1]))

        return result[1:]  # except for the input


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        for i in range(2):
            setattr(self, 'Scale_{}'.format(str(i)), PatchDiscriminator())
        self.n_D = 2

        print(self)
        print("the number of D parameters", sum(p.numel() for p in self.parameters() if p.requires_grad))

    def forward(self, x):
        result = []
        for i in range(self.n_D):
            result.append(getattr(self, 'Scale_{}'.format(i))(x))
            if i != self.n_D - 1:
                x = nn.AvgPool2d(kernel_size=3, padding=1, stride=2, count_include_pad=False)(x)
        return result


class Pix2pixHD_Trainer(Trainer):
    def training(self, net, datasets):
        """
          Args:
            net: (object) net & optimizer
            datasets : (list) [train, val] dataset object
        """
        args = self.args
        best_trn_perform, best_val_perform = -1, -1
        steps = len(datasets[0]) // args.batch_size
        if steps * args.batch_size < len(datasets[0]):
            steps += 1

        if args.trigger == 'epoch':
            args.epochs = args.terminal
            args.iters = steps * args.terminal
            args.iter_interval = steps * args.interval
        else:
            args.epochs = args.terminal // steps + 1
            args.iters = args.terminal
            args.iter_interval = args.interval

        net.train()
        D = Discriminator().cuda()
        optimizer_D = torch.optim.Adam(D.parameters(), lr=0.0002, betas=(0.5, 0.999), eps=1e-4)

        trn_loss, trn_acc = [], []
        start = time.time()
        criterion_pix2pixHD = losses.pix2pixHDLoss()
        criterion_MSE = losses.MSELoss()
        criterion_FM = losses.L1Loss()
        criterionA = losses.DSSIMLoss()
        criterionB = losses.L1Loss()
        criterionExtra = losses.VGG16Loss()
        latent_loss_weight = 0.25

        for epoch in range(1, args.epochs + 1):
            self.epoch = epoch
            # setup data loader
            data_loader = DataLoader(datasets[0], args.batch_size, num_workers=4,
                                     shuffle=True, pin_memory=True, )
            for idx, sample in enumerate(tqdm(data_loader)):
                self.iter += 1
                if self.iter > args.iters:
                    self.iter -= 1
                    break
                # get tensors from sample
                x = sample["src"]
                y = sample["tar"]
                if args.cuda:
                    x = x.cuda()
                    y = y.cuda()
                # forwading

                gen_l, gen_a, gen_b, dec, latent_loss = net(x)
                # gen_L, gen_a, gen_b, mu, logvar = net(x)
                # gen_y = torch.cat([gen_l, gen_ab], dim=1)
                gen_y = torch.cat([gen_l, gen_a, gen_b], dim=1)

                D_loss, target_tensor, generated_tensor = criterion_pix2pixHD(D, net, x, y)


                latent_loss = latent_loss.mean()
                # loss_l = self.criterion(gen_y[:,:1,:,:], y[:,:1,:,:])
                # loss_ab = self.criterion(gen_y[:,1:,:,:], y[:,1:,:,:])
                # loss_L = criterion_L(gen_y[:, 0, :, :], y[:, 0, :, :])

                loss_l = criterionA(gen_y[:, :1, :, :], y[:, :1, :, :])
                #loss_a = criterionB(gen_y[:, 1, :, :], y[:, 1, :, :])
                #loss_b = criterionB(gen_y[:, 2, :, :], y[:, 2, :, :])
                #loss_l = criterionA(gen_y[:, :1, :, :], y[:, :1, :, :])

                loss_a = criterionB(gen_y[:, 1, :, :], y[:, 1, :, :])
                loss_b = criterionB(gen_y[:, 2, :, :], y[:, 2, :, :])


                loss_VGG = criterionExtra(gen_y, y)
                # GAN Loss

                loss = loss_l + loss_a + loss_b + latent_loss_weight * latent_loss + args.alpha * loss_VGG
                # loss = loss_L + loss_a + loss_b + loss_VAE + args.alpha * loss_VGG


                # update parameters
                optimizer_D.zero_grad()
                D_loss.backward()
                optimizer_D.step()

                net.optimizer.zero_grad()
                loss.backward()
                net.optimizer.step()

                # update taining condition
                trn_loss.append(loss.detach().item())
                trn_acc.append(self.evaluator(gen_y.detach(), y.detach()).item())
                # validating
                if self.iter % args.iter_interval == 0:
                    trn_fps = (args.iter_interval * args.batch_size) / (time.time() - start)
                    self.trn_log = [round(sum(trn_loss) / len(trn_loss), 3),
                                    round(sum(trn_acc) / len(trn_acc), 3),
                                    round(trn_fps, 3)]

                    self.validating(net, datasets[1])
                    self.logging(verbose=True)
                    if self.val_log[1] >= best_val_perform:
                        best_trn_perform = self.trn_log[1]
                        best_val_perform = self.val_log[1]
                        checkpoint_info = [self.repr, self.epoch, self.iter,
                                           best_trn_perform, best_val_perform]

                        # save better checkpoint
                        self.save_checkpoint(net)
                    # reinitialize
                    start = time.time()
                    trn_loss, trn_acc = [], []
                    net.train()


        df = pd.DataFrame([checkpoint_info],
                          columns=["experiment", "best_epoch", "best_iter", self.header[3], self.header[6]])
        if os.path.exists(os.path.join(Checkpoint_DIR, 'checkpoint.csv')):
            prev_df = pd.read_csv(os.path.join(Checkpoint_DIR, 'checkpoint.csv'))
            df = prev_df.append(df)
        df.to_csv(os.path.join(Checkpoint_DIR, 'checkpoint.csv'), index=False, float_format='%.3f')

        print("Best {} Performance: \n".format(repr(self.evaluator)))
        print("\t Trn:", best_trn_perform)
        print("\t Val:", best_val_perform)

    def validating(self, net, dataset):
        """
          Args:
            net: (object) pytorch net
            batch_size: (int)
            dataset : (object) dataset
          return [loss, acc]
        """
        args = self.args
        data_loader = DataLoader(dataset, args.batch_size, num_workers=4,
                                 shuffle=False, pin_memory=True, )
        val_loss, val_acc = [], []
        start = time.time()
        net.eval()
        with torch.set_grad_enabled(False):
            for idx, sample in enumerate(data_loader):
                # get tensors from sample
                x = sample["src"]
                y = sample["tar"]
                if args.cuda:
                    x = x.cuda()
                    y = y.cuda()
                # forwading
                gen_l, gen_a, gen_b, _, _ = net(x)
                # gen_l, gen_ab = net(x)
                gen_y = torch.cat([gen_l, gen_a, gen_b], dim=1)
                # gen_y = torch.cat([gen_L, gen_a, gen_b], dim=1)
                val_loss.append(self.criterion(gen_y.detach(), y.detach()).item())
                val_acc.append(self.evaluator(gen_y.detach(), y.detach()).item())

        val_fps = (len(val_loss) * args.batch_size) / (time.time() - start)
        self.val_log = [round(sum(val_loss) / len(val_loss), 3),
                        round(sum(val_acc) / len(val_acc), 3),
                        round(val_fps, 3)]
        self.save_snapshot(x.detach(), y.detach(), gen_y.detach(), dataset)

class Pix2pixHDWGAN_Trainer(Trainer):
    def training(self, net, datasets):
        """
          Args:
            net: (object) net & optimizer
            datasets : (list) [train, val] dataset object
        """
        args = self.args
        best_trn_perform, best_val_perform = -1, -1
        steps = len(datasets[0]) // args.batch_size
        if steps * args.batch_size < len(datasets[0]):
            steps += 1

        if args.trigger == 'epoch':
            args.epochs = args.terminal
            args.iters = steps * args.terminal
            args.iter_interval = steps * args.interval
        else:
            args.epochs = args.terminal // steps + 1
            args.iters = args.terminal
            args.iter_interval = args.interval

        net.train()
        D = Discriminator().cuda()
        optimizer_D = torch.optim.Adam(D.parameters(), lr=0.0002, betas=(0.5, 0.999), eps=1e-4)

        trn_loss, trn_acc = [], []
        start = time.time()
        # criterion_pix2pixHD = losses.pix2pixHDLoss()
        criterion_pix2pixHDWGAN = losses.pix2pixHDWGANLoss()
        criterionA = losses.DSSIMLoss()
        criterionB = losses.L1Loss()
        criterionExtra = losses.VGG16Loss()
        latent_loss_weight = 0.25

        for epoch in range(1, args.epochs + 1):
            self.epoch = epoch
            # setup data loader
            data_loader = DataLoader(datasets[0], args.batch_size, num_workers=4,
                                     shuffle=True, pin_memory=True, )
            for idx, sample in enumerate(tqdm(data_loader)):
                self.iter += 1
                if self.iter > args.iters:
                    self.iter -= 1
                    break
                # get tensors from sample
                x = sample["src"]
                y = sample["tar"]
                if args.cuda:
                    x = x.cuda()
                    y = y.cuda()
                # forwading

                gen_l, gen_a, gen_b, dec, latent_loss = net(x)
                # gen_L, gen_a, gen_b, mu, logvar = net(x)
                # gen_y = torch.cat([gen_l, gen_ab], dim=1)
                gen_y = torch.cat([gen_l, gen_a, gen_b], dim=1)

                D_loss, target_tensor, generated_tensor = criterion_pix2pixHDWGAN(D, net, x, y)

                latent_loss = latent_loss.mean()
                # loss_l = self.criterion(gen_y[:,:1,:,:], y[:,:1,:,:])
                # loss_ab = self.criterion(gen_y[:,1:,:,:], y[:,1:,:,:])
                # loss_L = criterion_L(gen_y[:, 0, :, :], y[:, 0, :, :])

                loss_l = criterionA(gen_y[:, :1, :, :], y[:, :1, :, :])
                #loss_a = criterionB(gen_y[:, 1, :, :], y[:, 1, :, :])
                #loss_b = criterionB(gen_y[:, 2, :, :], y[:, 2, :, :])
                #loss_l = criterionA(gen_y[:, :1, :, :], y[:, :1, :, :])

                loss_a = criterionB(gen_y[:, 1, :, :], y[:, 1, :, :])
                loss_b = criterionB(gen_y[:, 2, :, :], y[:, 2, :, :])


                loss_VGG = criterionExtra(gen_y, y)
                # GAN Loss

                loss = loss_l + loss_a + loss_b + latent_loss_weight * latent_loss + args.alpha * loss_VGG
                # loss = loss_L + loss_a + loss_b + loss_VAE + args.alpha * loss_VGG
                # update parameters
                optimizer_D.zero_grad()
                D_loss.backward()
                optimizer_D.step()

                net.optimizer.zero_grad()
                loss.backward()
                net.optimizer.step()

                # update taining condition
                trn_loss.append(loss.detach().item())
                trn_acc.append(self.evaluator(gen_y.detach(), y.detach()).item())
                # validating
                if self.iter % args.iter_interval == 0:
                    trn_fps = (args.iter_interval * args.batch_size) / (time.time() - start)
                    self.trn_log = [round(sum(trn_loss) / len(trn_loss), 3),
                                    round(sum(trn_acc) / len(trn_acc), 3),
                                    round(trn_fps, 3)]

                    self.validating(net, datasets[1])
                    self.logging(verbose=True)
                    if self.val_log[1] >= best_val_perform:
                        best_trn_perform = self.trn_log[1]
                        best_val_perform = self.val_log[1]
                        checkpoint_info = [self.repr, self.epoch, self.iter,
                                           best_trn_perform, best_val_perform]

                        # save better checkpoint
                        self.save_checkpoint(net)
                    # reinitialize
                    start = time.time()
                    trn_loss, trn_acc = [], []
                    net.train()


        df = pd.DataFrame([checkpoint_info],
                          columns=["experiment", "best_epoch", "best_iter", self.header[3], self.header[6]])
        if os.path.exists(os.path.join(Checkpoint_DIR, 'checkpoint.csv')):
            prev_df = pd.read_csv(os.path.join(Checkpoint_DIR, 'checkpoint.csv'))
            df = prev_df.append(df)
        df.to_csv(os.path.join(Checkpoint_DIR, 'checkpoint.csv'), index=False, float_format='%.3f')

        print("Best {} Performance: \n".format(repr(self.evaluator)))
        print("\t Trn:", best_trn_perform)
        print("\t Val:", best_val_perform)

    def validating(self, net, dataset):
        """
          Args:
            net: (object) pytorch net
            batch_size: (int)
            dataset : (object) dataset
          return [loss, acc]
        """
        args = self.args
        data_loader = DataLoader(dataset, args.batch_size, num_workers=4,
                                 shuffle=False, pin_memory=True, )
        val_loss, val_acc = [], []
        start = time.time()
        net.eval()
        with torch.set_grad_enabled(False):
            for idx, sample in enumerate(data_loader):
                # get tensors from sample
                x = sample["src"]
                y = sample["tar"]
                if args.cuda:
                    x = x.cuda()
                    y = y.cuda()
                # forwading
                gen_l, gen_a, gen_b, _, _ = net(x)
                # gen_l, gen_ab = net(x)
                gen_y = torch.cat([gen_l, gen_a, gen_b], dim=1)
                # gen_y = torch.cat([gen_L, gen_a, gen_b], dim=1)
                val_loss.append(self.criterion(gen_y.detach(), y.detach()).item())
                val_acc.append(self.evaluator(gen_y.detach(), y.detach()).item())

        val_fps = (len(val_loss) * args.batch_size) / (time.time() - start)
        self.val_log = [round(sum(val_loss) / len(val_loss), 3),
                        round(sum(val_acc) / len(val_acc), 3),
                        round(val_fps, 3)]
        self.save_snapshot(x.detach(), y.detach(), gen_y.detach(), dataset)

class Pix2pixHDCustom_Trainer(Trainer):
    def training(self, net, datasets):
        """
          Args:
            net: (object) net & optimizer
            datasets : (list) [train, val] dataset object
        """
        args = self.args
        best_trn_perform, best_val_perform = -1, -1
        steps = len(datasets[0]) // args.batch_size
        if steps * args.batch_size < len(datasets[0]):
            steps += 1

        if args.trigger == 'epoch':
            args.epochs = args.terminal
            args.iters = steps * args.terminal
            args.iter_interval = steps * args.interval
        else:
            args.epochs = args.terminal // steps + 1
            args.iters = args.terminal
            args.iter_interval = args.interval

        net[0].train()
        net[1].train()
        net[2].train()
        D = Discriminator().cuda()
        optimizer_D = torch.optim.Adam(D.parameters(), lr=0.0002, betas=(0.5, 0.999), eps=1e-4)

        trn_loss, trn_acc = [], []
        start = time.time()
        # criterion_pix2pixHD = losses.pix2pixHDLoss()
        # criterion_pix2pixHDWGAN = losses.pix2pixHDWGANLoss()
        criterion_pix2pixHD_custom = losses.pix2pixHDLoss_custom()
        criterionA = losses.DSSIMLoss()
        criterionB = losses.L1Loss()
        criterionExtra = losses.VGG16Loss()
        latent_loss_weight = 0.25

        for epoch in range(1, args.epochs + 1):
            self.epoch = epoch
            # setup data loader
            data_loader = DataLoader(datasets[0], args.batch_size, num_workers=4,
                                     shuffle=True, pin_memory=True, )
            for idx, sample in enumerate(tqdm(data_loader)):
                self.iter += 1
                if self.iter > args.iters:
                    self.iter -= 1
                    break
                # get tensors from sample
                x = sample["src"]
                y = sample["tar"]
                if args.cuda:
                    x = x.cuda()
                    y = y.cuda()
                # forwading

                #gen_l, gen_a, gen_b, dec, latent_loss, x6 = net[0](x)
                #gen_l_edge, gen_a_edge, gen_b_edge, dec_edge, latent_loss_edge, x6_edge = net[1](x)
                #x6_cat = torch.cat([x6, x6_edge], dim=1)
                #gen_l_final, gen_a_final, gen_b_final = net[2](x6_cat)
                # gen_L, gen_a, gen_b, mu, logvar = net(x)
                # gen_y = torch.cat([gen_l, gen_ab], dim=1)
                #gen_y = torch.cat([gen_l, gen_a, gen_b], dim=1)
                #gen_y_edge = torch.cat([gen_l_edge, gen_a_edge, gen_b_edge], dim=1)
                #gen_y_final = torch.cat([gen_l_final, gen_a_final, gen_b_final], dim=1)
                # D_loss, target_tensor, generated_tensor = criterion_pix2pixHDWGAN(D, net, x, y)
                #D_loss, target_tensor, generated_tensor = criterion_pix2pixHD_custom(D, net[0], net[1], net[2], x, y)
                #latent_loss = latent_loss.mean()
                #latent_loss_edge = latent_loss_edge.mean()
                # loss_l = self.criterion(gen_y[:,:1,:,:], y[:,:1,:,:])
                # loss_ab = self.criterion(gen_y[:,1:,:,:], y[:,1:,:,:])
                # loss_L = criterion_L(gen_y[:, 0, :, :], y[:, 0, :, :])

                #loss_l = criterionA(gen_y[:, :1, :, :], y[:, :1, :, :])
                #loss_l_edge = criterionA(gen_y_edge[:, :1, :, :], y[:, :1, :, :])
                #loss_l_final = criterionA(gen_y_final[:, :1, :, :], y[:, :1, :, :])
                #loss_a = criterionB(gen_y[:, 1, :, :], y[:, 1, :, :])
                #loss_b = criterionB(gen_y[:, 2, :, :], y[:, 2, :, :])
                #loss_l = criterionA(gen_y[:, :1, :, :], y[:, :1, :, :])

                #loss_a = criterionB(gen_y[:, 1, :, :], y[:, 1, :, :])
                #loss_b = criterionB(gen_y[:, 2, :, :], y[:, 2, :, :])
                #loss_a_edge = criterionB(gen_y_edge[:, 1, :, :], y[:, 1, :, :])
                #loss_b_edge = criterionB(gen_y_edge[:, 2, :, :], y[:, 2, :, :])
                #loss_a_final = criterionB(gen_y_final[:, 1, :, :], y[:, 1, :, :])
                #loss_b_final = criterionB(gen_y_final[:, 2, :, :], y[:, 2, :, :])

                #loss_VGG = criterionExtra(gen_y, y)
                #loss_VGG_edge = criterionExtra(gen_y_edge, y)
                #loss_VGG_final = criterionExtra(gen_y_final, y)
                # GAN Loss

                #loss = loss_l + loss_a + loss_b + latent_loss_weight * latent_loss + args.alpha * loss_VGG
                #loss_edge = loss_l_edge + loss_a_edge + loss_b_edge + latent_loss_weight * latent_loss_edge + args.alpha * loss_VGG_edge
                #loss_dec = loss_l_final + loss_a_final + loss_b_final + args.alpha * loss_VGG_final
                # loss_final = loss + loss_edge + loss_dec
                # loss = loss_L + loss_a + loss_b + loss_VAE + args.alpha * loss_VGG
                # update parameters
                #optimizer_D.zero_grad()
                #D_loss.backward()
                #optimizer_D.step()

                net[0].optimizer.zero_grad()
                gen_l, gen_a, gen_b, dec, latent_loss = net[0](x)
                gen_y = torch.cat([gen_l, gen_a, gen_b], dim=1)
                latent_loss = latent_loss.mean()
                loss_l = criterionA(gen_y[:, :1, :, :], y[:, :1, :, :])
                loss_a = criterionB(gen_y[:, 1, :, :], y[:, 1, :, :])
                loss_b = criterionB(gen_y[:, 2, :, :], y[:, 2, :, :])
                loss_VGG = criterionExtra(gen_y, y)
                loss = loss_l + loss_a + loss_b + latent_loss_weight * latent_loss + args.alpha * loss_VGG
                loss.backward()
                net[0].optimizer.step()

                net[1].optimizer.zero_grad()
                gen_l_edge, gen_a_edge, gen_b_edge, dec_edge, latent_loss_edge = net[1](x)
                gen_y_edge = torch.cat([gen_l_edge, gen_a_edge, gen_b_edge], dim=1)
                latent_loss_edge = latent_loss_edge.mean()
                loss_l_edge = criterionA(gen_y_edge[:, :1, :, :], y[:, :1, :, :])
                loss_a_edge = criterionB(gen_y_edge[:, 1, :, :], y[:, 1, :, :])
                loss_b_edge = criterionB(gen_y_edge[:, 2, :, :], y[:, 2, :, :])
                loss_VGG_edge = criterionExtra(gen_y_edge, y)
                loss_edge = loss_l_edge + loss_a_edge + loss_b_edge + latent_loss_weight * latent_loss_edge + args.alpha * loss_VGG_edge
                loss_edge.backward()
                net[1].optimizer.step()

                net[2].optimizer.zero_grad()
                gen_y = gen_y.detach()
                gen_y_edge = gen_y_edge.detach()
                gen_input = torch.cat([gen_y, gen_y_edge], dim=1)
                gen_l_final, gen_a_final, gen_b_final, dec_final, latent_loss_final = net[2](gen_input)
                gen_y_final = torch.cat([gen_l_final, gen_a_final, gen_b_final], dim=1)
                latent_loss_final = latent_loss_final.mean()
                loss_l_final = criterionA(gen_y_final[:, :1, :, :], y[:, :1, :, :])
                loss_a_final = criterionB(gen_y_final[:, 1, :, :], y[:, 1, :, :])
                loss_b_final = criterionB(gen_y_final[:, 2, :, :], y[:, 2, :, :])
                loss_VGG_final = criterionExtra(gen_y_final, y)
                loss_dec = loss_l_final + loss_a_final + loss_b_final + latent_loss_weight * latent_loss_final + args.alpha * loss_VGG_final
                loss_dec.backward()
                net[2].optimizer.step()

                #net_edge.optimizer.zero_grad()
                #loss_edge.backward()
                #net_edge.optimizer.step()


                # update taining condition
                trn_loss.append(loss_dec.detach().item())
                trn_acc.append(self.evaluator(gen_y_final.detach(), y.detach()).item())
                # validating
                if self.iter % args.iter_interval == 0:
                    trn_fps = (args.iter_interval * args.batch_size) / (time.time() - start)
                    self.trn_log = [round(sum(trn_loss) / len(trn_loss), 3),
                                    round(sum(trn_acc) / len(trn_acc), 3),
                                    round(trn_fps, 3)]

                    self.validating(net, datasets[1])
                    self.logging(verbose=True)
                    if self.val_log[1] >= best_val_perform:
                        best_trn_perform = self.trn_log[1]
                        best_val_perform = self.val_log[1]
                        checkpoint_info = [self.repr, self.epoch, self.iter,
                                           best_trn_perform, best_val_perform]

                        # save better checkpoint
                        self.save_checkpoint_custom(net)
                    # reinitialize
                    start = time.time()
                    trn_loss, trn_acc = [], []
                    net[0].train()
                    net[1].train()
                    net[2].train()

        df = pd.DataFrame([checkpoint_info],
                          columns=["experiment", "best_epoch", "best_iter", self.header[3], self.header[6]])
        if os.path.exists(os.path.join(Checkpoint_DIR, 'checkpoint.csv')):
            prev_df = pd.read_csv(os.path.join(Checkpoint_DIR, 'checkpoint.csv'))
            df = prev_df.append(df)
        df.to_csv(os.path.join(Checkpoint_DIR, 'checkpoint.csv'), index=False, float_format='%.3f')

        print("Best {} Performance: \n".format(repr(self.evaluator)))
        print("\t Trn:", best_trn_perform)
        print("\t Val:", best_val_perform)

    def validating(self, net, dataset):
        """
          Args:
            net: (object) pytorch net
            batch_size: (int)
            dataset : (object) dataset
          return [loss, acc]
        """
        args = self.args
        data_loader = DataLoader(dataset, args.batch_size, num_workers=4,
                                 shuffle=False, pin_memory=True, )
        val_loss, val_acc = [], []
        start = time.time()
        net[0].eval()
        net[1].eval()
        net[2].eval()
        with torch.set_grad_enabled(False):
            for idx, sample in enumerate(data_loader):
                # get tensors from sample
                x = sample["src"]
                y = sample["tar"]
                if args.cuda:
                    x = x.cuda()
                    y = y.cuda()
                # forwading
                fake_L_G, fake_a_G, fake_b_G, _, _ = net[0](x)
                fake_L_edge, fake_a_edge, fake_b_edge, _, _ = net[1](x)
                fake_G = torch.cat([fake_L_G, fake_a_G, fake_b_G], dim=1)
                fake_edge = torch.cat([fake_L_edge, fake_a_edge, fake_b_edge], dim=1)
                fake_input = torch.cat([fake_G, fake_edge], dim=1)
                gen_l, gen_a, gen_b, _, _ = net[2](fake_input)
                # gen_l, gen_ab = net(x)
                gen_y = torch.cat([gen_l, gen_a, gen_b], dim=1)
                # gen_y = torch.cat([gen_L, gen_a, gen_b], dim=1)
                val_loss.append(self.criterion(gen_y.detach(), y.detach()).item())
                val_acc.append(self.evaluator(gen_y.detach(), y.detach()).item())

        val_fps = (len(val_loss) * args.batch_size) / (time.time() - start)
        self.val_log = [round(sum(val_loss) / len(val_loss), 3),
                        round(sum(val_acc) / len(val_acc), 3),
                        round(val_fps, 3)]
        self.save_snapshot(x.detach(), y.detach(), gen_y.detach(), dataset)

class Pix2pixHDfinal_Trainer(Trainer):
    def training(self, net, datasets):
        """
          Args:
            net: (object) net & optimizer
            datasets : (list) [train, val] dataset object
        """
        args = self.args
        best_trn_perform, best_val_perform = -1, -1
        steps = len(datasets[0]) // args.batch_size
        if steps * args.batch_size < len(datasets[0]):
            steps += 1

        if args.trigger == 'epoch':
            args.epochs = args.terminal
            args.iters = steps * args.terminal
            args.iter_interval = steps * args.interval
        else:
            args.epochs = args.terminal // steps + 1
            args.iters = args.terminal
            args.iter_interval = args.interval

        net[0].train()
        net[1].train()
        net[2].train()
        #D = Discriminator().cuda()
        #optimizer_D = torch.optim.Adam(D.parameters(), lr=0.0002, betas=(0.5, 0.999), eps=1e-4)

        trn_loss, trn_acc = [], []
        start = time.time()
        # criterion_pix2pixHD = losses.pix2pixHDLoss()
        # criterion_pix2pixHDWGAN = losses.pix2pixHDWGANLoss()
        #criterion_pix2pixHD_custom = losses.pix2pixHDLoss_custom()
        criterionA = losses.DSSIMLoss()
        criterionB = losses.L1Loss()
        criterionExtra = losses.VGG16Loss()
        for epoch in range(1, args.epochs + 1):
            self.epoch = epoch
            # setup data loader
            data_loader = DataLoader(datasets[0], args.batch_size, num_workers=4,
                                     shuffle=True, pin_memory=True, )
            for idx, sample in enumerate(tqdm(data_loader)):
                self.iter += 1
                if self.iter > args.iters:
                    self.iter -= 1
                    break
                # get tensors from sample
                x = sample["src"]
                y = sample["tar"]
                if args.cuda:
                    x = x.cuda()
                    y = y.cuda()
                # forwading


                #D_loss, target_tensor, generated_tensor = criterion_pix2pixHD_custom(D, net[0], net[1], net[2], x, y)

                # update parameters
                #optimizer_D.zero_grad()
                #D_loss.backward()
                #optimizer_D.step()

                net[0].optimizer.zero_grad()
                gen_l, gen_a, gen_b, x10_L, x10_a, x10_b = net[0](x)
                gen_y = torch.cat([gen_l, gen_a, gen_b], dim=1)
                loss_l = criterionA(gen_y[:, :1, :, :], y[:, :1, :, :])
                loss_a = criterionB(gen_y[:, 1, :, :], y[:, 1, :, :])
                loss_b = criterionB(gen_y[:, 2, :, :], y[:, 2, :, :])
                loss_VGG = criterionExtra(gen_y, y)
                loss = loss_l + loss_a + loss_b + args.alpha * loss_VGG
                loss.backward()
                net[0].optimizer.step()

                net[1].optimizer.zero_grad()
                gen_l_edge, gen_a_edge, gen_b_edge, x10_L_edge, x10_a_edge, x10_b_edge = net[1](x)
                gen_y_edge = torch.cat([gen_l_edge, gen_a_edge, gen_b_edge], dim=1)
                loss_l_edge = criterionA(gen_y_edge[:, :1, :, :], y[:, :1, :, :])
                loss_a_edge = criterionB(gen_y_edge[:, 1, :, :], y[:, 1, :, :])
                loss_b_edge = criterionB(gen_y_edge[:, 2, :, :], y[:, 2, :, :])
                loss_VGG_edge = criterionExtra(gen_y_edge, y)
                loss_edge = loss_l_edge + loss_a_edge + loss_b_edge + args.alpha * loss_VGG_edge

                loss_edge.backward()
                net[1].optimizer.step()

                net[2].optimizer.zero_grad()
                x10 = torch.cat([x10_L.detach(), x10_a.detach(), x10_b.detach()], dim=1)
                x10_edge = torch.cat([x10_L_edge.detach(), x10_a_edge.detach(), x10_b_edge.detach()], dim=1)
                x10_input = torch.cat([x10, x10_edge], dim=1)
                gen_l_final, gen_a_final, gen_b_final = net[2](x10_input)
                gen_y_final = torch.cat([gen_l_final, gen_a_final, gen_b_final], dim=1)
                loss_l_final = criterionA(gen_y_final[:, :1, :, :], y[:, :1, :, :])
                loss_a_final = criterionB(gen_y_final[:, 1, :, :], y[:, 1, :, :])
                loss_b_final = criterionB(gen_y_final[:, 2, :, :], y[:, 2, :, :])
                loss_VGG_final = criterionExtra(gen_y_final, y)
                loss_final = loss_l_final + loss_a_final + loss_b_final + args.alpha * loss_VGG_final
                loss_final.backward()
                net[2].optimizer.step()

                #net_edge.optimizer.zero_grad()
                #loss_edge.backward()
                #net_edge.optimizer.step()


                # update taining condition
                trn_loss.append(loss_final.detach().item())
                trn_acc.append(self.evaluator(gen_y_final.detach(), y.detach()).item())
                # validating
                if self.iter % args.iter_interval == 0:
                    trn_fps = (args.iter_interval * args.batch_size) / (time.time() - start)
                    self.trn_log = [round(sum(trn_loss) / len(trn_loss), 3),
                                    round(sum(trn_acc) / len(trn_acc), 3),
                                    round(trn_fps, 3)]

                    self.validating(net, datasets[1])
                    self.logging(verbose=True)
                    if self.val_log[1] >= best_val_perform:
                        best_trn_perform = self.trn_log[1]
                        best_val_perform = self.val_log[1]
                        checkpoint_info = [self.repr, self.epoch, self.iter,
                                           best_trn_perform, best_val_perform]

                        # save better checkpoint
                        self.save_checkpoint_custom(net)
                    # reinitialize
                    start = time.time()
                    trn_loss, trn_acc = [], []
                    net[0].train()
                    net[1].train()
                    net[2].train()

        df = pd.DataFrame([checkpoint_info],
                          columns=["experiment", "best_epoch", "best_iter", self.header[3], self.header[6]])
        if os.path.exists(os.path.join(Checkpoint_DIR, 'checkpoint.csv')):
            prev_df = pd.read_csv(os.path.join(Checkpoint_DIR, 'checkpoint.csv'))
            df = prev_df.append(df)
        df.to_csv(os.path.join(Checkpoint_DIR, 'checkpoint.csv'), index=False, float_format='%.3f')

        print("Best {} Performance: \n".format(repr(self.evaluator)))
        print("\t Trn:", best_trn_perform)
        print("\t Val:", best_val_perform)

    def validating(self, net, dataset):
        """
          Args:
            net: (object) pytorch net
            batch_size: (int)
            dataset : (object) dataset
          return [loss, acc]
        """
        args = self.args
        data_loader = DataLoader(dataset, args.batch_size, num_workers=4,
                                 shuffle=False, pin_memory=True, )
        val_loss, val_acc = [], []
        start = time.time()
        net[0].eval()
        net[1].eval()
        net[2].eval()
        with torch.set_grad_enabled(False):
            for idx, sample in enumerate(data_loader):
                # get tensors from sample
                x = sample["src"]
                y = sample["tar"]
                if args.cuda:
                    x = x.cuda()
                    y = y.cuda()
                # forwading
                _, _, _, x10_L, x10_a, x10_b = net[0](x)
                _, _, _, x10_L_edge, x10_a_edge, x10_b_edge = net[1](x)
                x10 = torch.cat([x10_L.detach(), x10_a.detach(), x10_b.detach()], dim=1)
                x10_edge = torch.cat([x10_L_edge.detach(), x10_a_edge.detach(), x10_b.detach()], dim=1)
                x10_input = torch.cat([x10, x10_edge], dim=1)
                gen_l, gen_a, gen_b = net[2](x10_input)
                # gen_l, gen_ab = net(x)
                gen_y = torch.cat([gen_l, gen_a, gen_b], dim=1)
                # gen_y = torch.cat([gen_L, gen_a, gen_b], dim=1)
                val_loss.append(self.criterion(gen_y.detach(), y.detach()).item())
                val_acc.append(self.evaluator(gen_y.detach(), y.detach()).item())

        val_fps = (len(val_loss) * args.batch_size) / (time.time() - start)
        self.val_log = [round(sum(val_loss) / len(val_loss), 3),
                        round(sum(val_acc) / len(val_acc), 3),
                        round(val_fps, 3)]
        self.save_snapshot(x.detach(), y.detach(), gen_y.detach(), dataset)

class Pix2pixHDalpha_Trainer(Trainer):
    def training(self, net, datasets):
        """
          Args:
            net: (object) net & optimizer
            datasets : (list) [train, val] dataset object
        """
        args = self.args
        best_trn_perform, best_val_perform = -1, -1
        steps = len(datasets[0]) // args.batch_size
        if steps * args.batch_size < len(datasets[0]):
            steps += 1

        if args.trigger == 'epoch':
            args.epochs = args.terminal
            args.iters = steps * args.terminal
            args.iter_interval = steps * args.interval
        else:
            args.epochs = args.terminal // steps + 1
            args.iters = args.terminal
            args.iter_interval = args.interval

        net[0].train()
        net[1].train()
        net[2].train()
        D = Discriminator().cuda()
        optimizer_D = torch.optim.Adam(D.parameters(), lr=0.0002, betas=(0.5, 0.999), eps=1e-4)

        trn_loss, trn_acc = [], []
        start = time.time()
        # criterion_pix2pixHD = losses.pix2pixHDLoss()
        # criterion_pix2pixHDWGAN = losses.pix2pixHDWGANLoss()
        criterion_pix2pixHD_custom = losses.pix2pixHDLoss_custom()
        criterionA = losses.DSSIMLoss()
        criterionB = losses.L1Loss()
        criterionExtra = losses.VGG16Loss()
        for epoch in range(1, args.epochs + 1):
            self.epoch = epoch
            # setup data loader
            data_loader = DataLoader(datasets[0], args.batch_size, num_workers=4,
                                     shuffle=True, pin_memory=True, )
            for idx, sample in enumerate(tqdm(data_loader)):
                self.iter += 1
                if self.iter > args.iters:
                    self.iter -= 1
                    break
                # get tensors from sample
                x = sample["src"]
                y = sample["tar"]
                if args.cuda:
                    x = x.cuda()
                    y = y.cuda()
                # forwading


                D_loss, target_tensor, generated_tensor = criterion_pix2pixHD_custom(D, net[0], net[1], net[2], x, y)

                # update parameters
                optimizer_D.zero_grad()
                D_loss.backward()
                optimizer_D.step()

                net[0].optimizer.zero_grad()
                gen_l, gen_a, gen_b = net[0](x)
                gen_y = torch.cat([gen_l, gen_a, gen_b], dim=1)
                loss_l = criterionA(gen_y[:, :1, :, :], y[:, :1, :, :])
                loss_a = criterionB(gen_y[:, 1, :, :], y[:, 1, :, :])
                loss_b = criterionB(gen_y[:, 2, :, :], y[:, 2, :, :])
                loss_VGG = criterionExtra(gen_y, y)
                loss = loss_l + loss_a + loss_b + args.alpha * loss_VGG
                loss.backward()
                net[0].optimizer.step()

                net[1].optimizer.zero_grad()
                gen_l_edge, gen_a_edge, gen_b_edge = net[1](x)
                gen_y_edge = torch.cat([gen_l_edge, gen_a_edge, gen_b_edge], dim=1)
                loss_l_edge = criterionA(gen_y_edge[:, :1, :, :], y[:, :1, :, :])
                loss_a_edge = criterionB(gen_y_edge[:, 1, :, :], y[:, 1, :, :])
                loss_b_edge = criterionB(gen_y_edge[:, 2, :, :], y[:, 2, :, :])
                loss_VGG_edge = criterionExtra(gen_y_edge, y)
                loss_edge = loss_l_edge + loss_a_edge + loss_b_edge + args.alpha * loss_VGG_edge

                loss_edge.backward()
                net[1].optimizer.step()

                net[2].optimizer.zero_grad()
                gen_input = torch.cat([gen_y.detach(), gen_y_edge.detach()], dim=1)
                gen_l_final, gen_a_final, gen_b_final = net[2](gen_input)
                gen_y_final = torch.cat([gen_l_final, gen_a_final, gen_b_final], dim=1)
                loss_l_final = criterionA(gen_y_final[:, :1, :, :], y[:, :1, :, :])
                loss_a_final = criterionB(gen_y_final[:, 1, :, :], y[:, 1, :, :])
                loss_b_final = criterionB(gen_y_final[:, 2, :, :], y[:, 2, :, :])
                loss_VGG_final = criterionExtra(gen_y_final, y)
                loss_final = loss_l_final + loss_a_final + loss_b_final + args.alpha * loss_VGG_final
                loss_final.backward()
                net[2].optimizer.step()

                #net_edge.optimizer.zero_grad()
                #loss_edge.backward()
                #net_edge.optimizer.step()


                # update taining condition
                trn_loss.append(loss_final.detach().item())
                trn_acc.append(self.evaluator(gen_y_final.detach(), y.detach()).item())
                # validating
                if self.iter % args.iter_interval == 0:
                    trn_fps = (args.iter_interval * args.batch_size) / (time.time() - start)
                    self.trn_log = [round(sum(trn_loss) / len(trn_loss), 3),
                                    round(sum(trn_acc) / len(trn_acc), 3),
                                    round(trn_fps, 3)]

                    self.validating(net, datasets[1])
                    self.logging(verbose=True)
                    if self.val_log[1] >= best_val_perform:
                        best_trn_perform = self.trn_log[1]
                        best_val_perform = self.val_log[1]
                        checkpoint_info = [self.repr, self.epoch, self.iter,
                                           best_trn_perform, best_val_perform]

                        # save better checkpoint
                        self.save_checkpoint_custom(net)
                    # reinitialize
                    start = time.time()
                    trn_loss, trn_acc = [], []
                    net[0].train()
                    net[1].train()
                    net[2].train()

        df = pd.DataFrame([checkpoint_info],
                          columns=["experiment", "best_epoch", "best_iter", self.header[3], self.header[6]])
        if os.path.exists(os.path.join(Checkpoint_DIR, 'checkpoint.csv')):
            prev_df = pd.read_csv(os.path.join(Checkpoint_DIR, 'checkpoint.csv'))
            df = prev_df.append(df)
        df.to_csv(os.path.join(Checkpoint_DIR, 'checkpoint.csv'), index=False, float_format='%.3f')

        print("Best {} Performance: \n".format(repr(self.evaluator)))
        print("\t Trn:", best_trn_perform)
        print("\t Val:", best_val_perform)

    def validating(self, net, dataset):
        """
          Args:
            net: (object) pytorch net
            batch_size: (int)
            dataset : (object) dataset
          return [loss, acc]
        """
        args = self.args
        data_loader = DataLoader(dataset, args.batch_size, num_workers=4,
                                 shuffle=False, pin_memory=True, )
        val_loss, val_acc = [], []
        start = time.time()
        net[0].eval()
        net[1].eval()
        net[2].eval()
        with torch.set_grad_enabled(False):
            for idx, sample in enumerate(data_loader):
                # get tensors from sample
                x = sample["src"]
                y = sample["tar"]
                if args.cuda:
                    x = x.cuda()
                    y = y.cuda()
                # forwading
                gen_L_G, gen_a_G, gen_b_G = net[0](x)
                gen_L_e, gen_a_e, gen_b_e = net[1](x)
                gen_G = torch.cat([gen_L_G, gen_a_G, gen_b_G], dim=1)
                gen_e = torch.cat([gen_L_e, gen_a_e, gen_b_e], dim=1)
                gen_input = torch.cat([gen_G.detach(), gen_e.detach()], dim=1)
                gen_l, gen_a, gen_b = net[2](gen_input)
                # gen_l, gen_ab = net(x)
                gen_y = torch.cat([gen_l, gen_a, gen_b], dim=1)
                # gen_y = torch.cat([gen_L, gen_a, gen_b], dim=1)
                val_loss.append(self.criterion(gen_y.detach(), y.detach()).item())
                val_acc.append(self.evaluator(gen_y.detach(), y.detach()).item())

        val_fps = (len(val_loss) * args.batch_size) / (time.time() - start)
        self.val_log = [round(sum(val_loss) / len(val_loss), 3),
                        round(sum(val_acc) / len(val_acc), 3),
                        round(val_fps, 3)]
        self.save_snapshot(x.detach(), y.detach(), gen_y.detach(), dataset)

class beta_Trainer(Trainer):
    def training(self, net, datasets):
        """
          Args:
            net: (object) net & optimizer
            datasets : (list) [train, val] dataset object
        """
        args = self.args
        best_trn_perform, best_val_perform = -1, -1
        steps = len(datasets[0]) // args.batch_size
        if steps * args.batch_size < len(datasets[0]):
            steps += 1

        if args.trigger == 'epoch':
            args.epochs = args.terminal
            args.iters = steps * args.terminal
            args.iter_interval = steps * args.interval
        else:
            args.epochs = args.terminal // steps + 1
            args.iters = args.terminal
            args.iter_interval = args.interval

        net[0].train()
        net[1].train()
        net[2].train()
        #D = Discriminator().cuda()
        #optimizer_D = torch.optim.Adam(D.parameters(), lr=0.0002, betas=(0.5, 0.999), eps=1e-4)

        trn_loss, trn_acc = [], []
        start = time.time()
        # criterion_pix2pixHD = losses.pix2pixHDLoss()
        # criterion_pix2pixHDWGAN = losses.pix2pixHDWGANLoss()
        # criterion_pix2pixHD_custom = losses.pix2pixHDLoss_custom()
        criterionA = losses.DSSIMLoss()
        criterionB = losses.L1Loss()
        criterionExtra = losses.VGG16Loss()
        for epoch in range(1, args.epochs + 1):
            self.epoch = epoch
            # setup data loader
            data_loader = DataLoader(datasets[0], args.batch_size, num_workers=4,
                                     shuffle=True, pin_memory=True, )
            for idx, sample in enumerate(tqdm(data_loader)):
                self.iter += 1
                if self.iter > args.iters:
                    self.iter -= 1
                    break
                # get tensors from sample
                x = sample["src"]
                y = sample["tar"]
                if args.cuda:
                    x = x.cuda()
                    y = y.cuda()
                # forwading


                #D_loss, target_tensor, generated_tensor = criterion_pix2pixHD_custom(D, net[0], net[1], net[2], x, y)

                # update parameters
                #optimizer_D.zero_grad()
                #D_loss.backward()
                #optimizer_D.step()

                net[0].optimizer.zero_grad()
                gen_l_i, gen_ab_i = net[0](x)
                gen_y_i = torch.cat([gen_l_i, gen_ab_i], dim=1)
                loss_l_i = criterionA(gen_y_i[:, :1, :, :], y[:, :1, :, :])
                loss_ab_i = criterionB(gen_y_i[:,1:,:,:], y[:,1:,:,:])
                loss_VGG_i = criterionExtra(gen_y_i, y)
                loss_i = loss_l_i + loss_ab_i + args.alpha * loss_VGG_i
                loss_i.backward()
                net[0].optimizer.step()

                net[1].optimizer.zero_grad()
                gen_l_e, gen_ab_e = net[1](x)
                gen_y_e = torch.cat([gen_l_e, gen_ab_e], dim=1)
                loss_l_e = criterionA(gen_y_e[:, :1, :, :], y[:, :1, :, :])
                loss_ab_e = criterionB(gen_y_e[:,1:,:,:], y[:,1:,:,:])
                loss_VGG_e = criterionExtra(gen_y_e, y)
                loss_e = loss_l_e + loss_ab_e + args.alpha * loss_VGG_e
                loss_e.backward()
                net[1].optimizer.step()

                net[2].optimizer.zero_grad()
                #gen_input = torch.cat([gen_y_i.detach(), gen_y_e.detach()], dim=1)
                gen_y_i = gen_y_i.detach()
                gen_y_e = gen_y_e.detach()
                gen_l_final, gen_ab_final = net[2](gen_y_i, gen_y_e)
                gen_y_final = torch.cat([gen_l_final, gen_ab_final], dim=1)
                loss_l_final = criterionA(gen_y_final[:, :1, :, :], y[:, :1, :, :])
                loss_ab_final = criterionB(gen_y_final[:,1:,:,:], y[:,1:,:,:])
                loss_VGG_final = criterionExtra(gen_y_final, y)
                loss_final = loss_l_final + loss_ab_final + args.alpha * loss_VGG_final
                loss_final.backward()
                net[2].optimizer.step()

                #net_edge.optimizer.zero_grad()
                #loss_edge.backward()
                #net_edge.optimizer.step()


                # update taining condition
                trn_loss.append(loss_final.detach().item())
                trn_acc.append(self.evaluator(gen_y_final.detach(), y.detach()).item())
                # validating
                if self.iter % args.iter_interval == 0:
                    trn_fps = (args.iter_interval * args.batch_size) / (time.time() - start)
                    self.trn_log = [round(sum(trn_loss) / len(trn_loss), 3),
                                    round(sum(trn_acc) / len(trn_acc), 3),
                                    round(trn_fps, 3)]

                    self.validating(net, datasets[1])
                    self.logging(verbose=True)
                    if self.val_log[1] >= best_val_perform:
                        best_trn_perform = self.trn_log[1]
                        best_val_perform = self.val_log[1]
                        checkpoint_info = [self.repr, self.epoch, self.iter,
                                           best_trn_perform, best_val_perform]

                        # save better checkpoint
                        self.save_checkpoint_custom(net)
                    # reinitialize
                    start = time.time()
                    trn_loss, trn_acc = [], []
                    net[0].train()
                    net[1].train()
                    net[2].train()

        df = pd.DataFrame([checkpoint_info],
                          columns=["experiment", "best_epoch", "best_iter", self.header[3], self.header[6]])
        if os.path.exists(os.path.join(Checkpoint_DIR, 'checkpoint.csv')):
            prev_df = pd.read_csv(os.path.join(Checkpoint_DIR, 'checkpoint.csv'))
            df = prev_df.append(df)
        df.to_csv(os.path.join(Checkpoint_DIR, 'checkpoint.csv'), index=False, float_format='%.3f')

        print("Best {} Performance: \n".format(repr(self.evaluator)))
        print("\t Trn:", best_trn_perform)
        print("\t Val:", best_val_perform)

    def validating(self, net, dataset):
        """
          Args:
            net: (object) pytorch net
            batch_size: (int)
            dataset : (object) dataset
          return [loss, acc]
        """
        args = self.args
        data_loader = DataLoader(dataset, args.batch_size, num_workers=4,
                                 shuffle=False, pin_memory=True, )
        val_loss, val_acc = [], []
        start = time.time()
        net[0].eval()
        net[1].eval()
        net[2].eval()
        with torch.set_grad_enabled(False):
            for idx, sample in enumerate(data_loader):
                # get tensors from sample
                x = sample["src"]
                y = sample["tar"]
                if args.cuda:
                    x = x.cuda()
                    y = y.cuda()
                # forwading
                gen_L_G, gen_ab_G = net[0](x)
                gen_L_e, gen_ab_e = net[1](x)
                gen_G = torch.cat([gen_L_G, gen_ab_G], dim=1)
                gen_e = torch.cat([gen_L_e, gen_ab_e], dim=1)
                gen_G = gen_G.detach()
                gen_e = gen_e.detach()
                gen_l, gen_ab = net[2](gen_G, gen_e)
                # gen_l, gen_ab = net(x)
                gen_y = torch.cat([gen_l, gen_ab], dim=1)
                # gen_y = torch.cat([gen_L, gen_a, gen_b], dim=1)
                val_loss.append(self.criterion(gen_y.detach(), y.detach()).item())
                val_acc.append(self.evaluator(gen_y.detach(), y.detach()).item())

        val_fps = (len(val_loss) * args.batch_size) / (time.time() - start)
        self.val_log = [round(sum(val_loss) / len(val_loss), 3),
                        round(sum(val_acc) / len(val_acc), 3),
                        round(val_fps, 3)]
        self.save_snapshot(x.detach(), y.detach(), gen_y.detach(), dataset)

class delta_Trainer(Trainer):
    def training(self, net, datasets):
        """
          Args:
            net: (object) net & optimizer
            datasets : (list) [train, val] dataset object
        """
        args = self.args
        best_trn_perform, best_val_perform = -1, -1
        steps = len(datasets[0]) // args.batch_size
        if steps * args.batch_size < len(datasets[0]):
            steps += 1

        if args.trigger == 'epoch':
            args.epochs = args.terminal
            args.iters = steps * args.terminal
            args.iter_interval = steps * args.interval
        else:
            args.epochs = args.terminal // steps + 1
            args.iters = args.terminal
            args.iter_interval = args.interval

        net[0].train()
        net[1].train()
        net[2].train()
        D = Discriminator().cuda()
        optimizer_D = torch.optim.RMSprop(D.parameters(), lr=0.0002)

        trn_loss, trn_acc = [], []
        start = time.time()
        criterion_pix2pixHD = losses.pix2pixHDWGANLoss_3gen_FM_Lab()
        # criterion_pix2pixHDWGAN = losses.pix2pixHDWGANLoss()
        # criterion_pix2pixHD_custom = losses.pix2pixHDLoss_custom()
        criterionA = losses.DSSIMLoss()
        criterionB = losses.L1Loss()
        criterionExtra = losses.VGG16Loss()
        for epoch in range(1, args.epochs + 1):
            self.epoch = epoch
            # setup data loader
            data_loader = DataLoader(datasets[0], args.batch_size, num_workers=4,
                                     shuffle=True, pin_memory=True, )
            for idx, sample in enumerate(tqdm(data_loader)):
                self.iter += 1
                if self.iter > args.iters:
                    self.iter -= 1
                    break
                # get tensors from sample
                x = sample["src"]
                y = sample["tar"]
                if args.cuda:
                    x = x.cuda()
                    y = y.cuda()
                # forwading


                D_loss, G_loss, target_tensor, generated_tensor = criterion_pix2pixHD(D, net[0], net[1], net[2], x, y)

                # update parameters
                optimizer_D.zero_grad()
                for p in D.parameters():
                    p.data.clamp_(-0.01, 0.01)

                D_loss.backward()
                optimizer_D.step()

                net[2].optimizer.zero_grad()
                G_loss.backward()
                net[2].optimizer.step()

                net[0].optimizer.zero_grad()
                gen_l_i, gen_ab_i = net[0](x)
                gen_y_i = torch.cat([gen_l_i, gen_ab_i], dim=1)
                loss_l_i = criterionA(gen_y_i[:, :1, :, :], y[:, :1, :, :])
                loss_ab_i = criterionB(gen_y_i[:,1:,:,:], y[:,1:,:,:])
                loss_VGG_i = criterionExtra(gen_y_i, y)
                loss_i = loss_l_i + loss_ab_i + args.alpha * loss_VGG_i
                loss_i.backward()
                net[0].optimizer.step()

                net[1].optimizer.zero_grad()
                gen_l_e, gen_ab_e = net[1](x)
                gen_y_e = torch.cat([gen_l_e, gen_ab_e], dim=1)
                loss_l_e = criterionA(gen_y_e[:, :1, :, :], y[:, :1, :, :])
                loss_ab_e = criterionB(gen_y_e[:,1:,:,:], y[:,1:,:,:])
                loss_VGG_e = criterionExtra(gen_y_e, y)
                loss_e = loss_l_e + loss_ab_e + args.alpha * loss_VGG_e
                loss_e.backward()
                net[1].optimizer.step()

                net[2].optimizer.zero_grad()
                #gen_input = torch.cat([gen_y_i.detach(), gen_y_e.detach()], dim=1)
                gen_y_i = gen_y_i.detach()
                gen_y_e = gen_y_e.detach()
                gen_l_final, gen_ab_final = net[2](gen_y_i, gen_y_e)
                gen_y_final = torch.cat([gen_l_final, gen_ab_final], dim=1)
                loss_l_final = criterionA(gen_y_final[:, :1, :, :], y[:, :1, :, :])
                loss_ab_final = criterionB(gen_y_final[:,1:,:,:], y[:,1:,:,:])
                loss_VGG_final = criterionExtra(gen_y_final, y)
                loss_final = loss_l_final + loss_ab_final + args.alpha * loss_VGG_final
                loss_final.backward()
                net[2].optimizer.step()

                #net_edge.optimizer.zero_grad()
                #loss_edge.backward()
                #net_edge.optimizer.step()


                # update taining condition
                trn_loss.append(loss_final.detach().item())
                trn_acc.append(self.evaluator(gen_y_final.detach(), y.detach()).item())
                # validating
                if self.iter % args.iter_interval == 0:
                    trn_fps = (args.iter_interval * args.batch_size) / (time.time() - start)
                    self.trn_log = [round(sum(trn_loss) / len(trn_loss), 3),
                                    round(sum(trn_acc) / len(trn_acc), 3),
                                    round(trn_fps, 3)]

                    self.validating(net, datasets[1])
                    self.logging(verbose=True)
                    if self.val_log[1] >= best_val_perform:
                        best_trn_perform = self.trn_log[1]
                        best_val_perform = self.val_log[1]
                        checkpoint_info = [self.repr, self.epoch, self.iter,
                                           best_trn_perform, best_val_perform]

                        # save better checkpoint
                        self.save_checkpoint_custom(net)
                    # reinitialize
                    start = time.time()
                    trn_loss, trn_acc = [], []
                    net[0].train()
                    net[1].train()
                    net[2].train()

        df = pd.DataFrame([checkpoint_info],
                          columns=["experiment", "best_epoch", "best_iter", self.header[3], self.header[6]])
        if os.path.exists(os.path.join(Checkpoint_DIR, 'checkpoint.csv')):
            prev_df = pd.read_csv(os.path.join(Checkpoint_DIR, 'checkpoint.csv'))
            df = prev_df.append(df)
        df.to_csv(os.path.join(Checkpoint_DIR, 'checkpoint.csv'), index=False, float_format='%.3f')

        print("Best {} Performance: \n".format(repr(self.evaluator)))
        print("\t Trn:", best_trn_perform)
        print("\t Val:", best_val_perform)

    def validating(self, net, dataset):
        """
          Args:
            net: (object) pytorch net
            batch_size: (int)
            dataset : (object) dataset
          return [loss, acc]
        """
        args = self.args
        data_loader = DataLoader(dataset, args.batch_size, num_workers=4,
                                 shuffle=False, pin_memory=True, )
        val_loss, val_acc = [], []
        start = time.time()
        net[0].eval()
        net[1].eval()
        net[2].eval()
        with torch.set_grad_enabled(False):
            for idx, sample in enumerate(data_loader):
                # get tensors from sample
                x = sample["src"]
                y = sample["tar"]
                if args.cuda:
                    x = x.cuda()
                    y = y.cuda()
                # forwading
                gen_L_G, gen_ab_G = net[0](x)
                gen_L_e, gen_ab_e = net[1](x)
                gen_G = torch.cat([gen_L_G, gen_ab_G], dim=1)
                gen_e = torch.cat([gen_L_e, gen_ab_e], dim=1)
                gen_G = gen_G.detach()
                gen_e = gen_e.detach()
                gen_l, gen_ab = net[2](gen_G, gen_e)
                # gen_l, gen_ab = net(x)
                gen_y = torch.cat([gen_l, gen_ab], dim=1)
                # gen_y = torch.cat([gen_L, gen_a, gen_b], dim=1)
                val_loss.append(self.criterion(gen_y.detach(), y.detach()).item())
                val_acc.append(self.evaluator(gen_y.detach(), y.detach()).item())

        val_fps = (len(val_loss) * args.batch_size) / (time.time() - start)
        self.val_log = [round(sum(val_loss) / len(val_loss), 3),
                        round(sum(val_acc) / len(val_acc), 3),
                        round(val_fps, 3)]
        self.save_snapshot(x.detach(), y.detach(), gen_y.detach(), dataset)

class yTrainer(Trainer):
    def training(self, net, datasets):
        """
          Args:
            net: (object) net & optimizer
            datasets : (list) [train, val] dataset object
        """
        args = self.args
        best_trn_perform, best_val_perform = -1, -1
        steps = len(datasets[0]) // args.batch_size
        if steps * args.batch_size < len(datasets[0]):
            steps += 1

        if args.trigger == 'epoch':
            args.epochs = args.terminal
            args.iters = steps * args.terminal
            args.iter_interval = steps * args.interval
        else:
            args.epochs = args.terminal // steps + 1
            args.iters = args.terminal
            args.iter_interval = args.interval

        net.train()
        trn_loss, trn_acc = [], []
        start = time.time()
        criterionA = losses.DSSIMLoss()
        criterionB = losses.L1Loss()
        criterionExtra = losses.VGG16Loss()
        for epoch in range(1, args.epochs + 1):
            self.epoch = epoch
            # setup data loader
            data_loader = DataLoader(datasets[0], args.batch_size, num_workers=4,
                                     shuffle=True, pin_memory=True,)
            for idx, sample in enumerate(data_loader):
                self.iter += 1
                if self.iter > args.iters:
                    self.iter -= 1
                    break
                # get tensors from sample
                x = sample["src"]
                y = sample["tar"]
                if args.cuda:
                    x = x.cuda()
                    y = y.cuda()
                # forwading
                gen_l, gen_ab = net(x)
                gen_y = torch.cat([gen_l, gen_ab], dim=1)
                # loss_l = self.criterion(gen_y[:,:1,:,:], y[:,:1,:,:])
                # loss_ab = self.criterion(gen_y[:,1:,:,:], y[:,1:,:,:])
                loss_l = criterionA(gen_y[:,:1,:,:], y[:,:1,:,:])
                loss_ab = criterionB(gen_y[:,1:,:,:], y[:,1:,:,:])
                loss_lab = criterionExtra(gen_y, y)
                loss = loss_l + loss_ab + args.alpha * loss_lab
                # update parameters
                net.optimizer.zero_grad()
                loss.backward()
                net.optimizer.step()
                # update taining condition
                trn_loss.append(loss.detach().item())
                trn_acc.append(self.evaluator(gen_y.detach(), y.detach()).item())
                # validating
                if self.iter % args.iter_interval == 0:
                    trn_fps = (args.iter_interval * args.batch_size) / (time.time() - start)
                    self.trn_log = [round(sum(trn_loss) / len(trn_loss), 3), 
                                    round(sum(trn_acc) / len(trn_acc), 3),
                                    round(trn_fps, 3)]
 
                    self.validating(net, datasets[1])
                    self.logging(verbose=True)
                    if self.val_log[1] >= best_val_perform:
                        best_trn_perform = self.trn_log[1]
                        best_val_perform = self.val_log[1]
                        checkpoint_info = [self.repr, self.epoch, self.iter,
                                           best_trn_perform, best_val_perform]
                        # save better checkpoint
                        self.save_checkpoint(net)
                    # reinitialize
                    start = time.time()
                    trn_loss, trn_acc = [], []
                    net.train()

        df = pd.DataFrame([checkpoint_info],
                          columns=["experiment", "best_epoch", "best_iter", self.header[3], self.header[6]])
        if os.path.exists(os.path.join(Checkpoint_DIR, 'checkpoint.csv')):
            prev_df = pd.read_csv(os.path.join(Checkpoint_DIR, 'checkpoint.csv'))
            df = prev_df.append(df)
        df.to_csv(os.path.join(Checkpoint_DIR, 'checkpoint.csv'), index=False,  float_format='%.3f')

        print("Best {} Performance: \n".format(repr(self.evaluator)))
        print("\t Trn:", best_trn_perform)
        print("\t Val:", best_val_perform)

    def validating(self, net, dataset):
        """
          Args:
            net: (object) pytorch net
            batch_size: (int)
            dataset : (object) dataset
          return [loss, acc]
        """
        args = self.args
        data_loader = DataLoader(dataset, args.batch_size, num_workers=4,
                                 shuffle=False, pin_memory=True,)
        val_loss, val_acc = [], []
        start = time.time()
        net.eval()
        with torch.set_grad_enabled(False):
            for idx, sample in enumerate(data_loader):
                # get tensors from sample
                x = sample["src"]
                y = sample["tar"]
                if args.cuda:
                    x = x.cuda()
                    y = y.cuda()
                # forwading
                gen_l, gen_ab = net(x)
                gen_y = torch.cat([gen_l, gen_ab], dim=1)
                val_loss.append(self.criterion(gen_y.detach(), y.detach()).item())
                val_acc.append(self.evaluator(gen_y.detach(), y.detach()).item())

        val_fps = (len(val_loss) * args.batch_size) / (time.time() - start)
        self.val_log = [round(sum(val_loss) / len(val_loss), 3), 
                        round(sum(val_acc) / len(val_acc), 3),
                        round(val_fps, 3)]
        self.save_snapshot(x.detach(), y.detach(), gen_y.detach(), dataset)


class yTrainer3D(Trainer):
    def training(self, net, datasets):
        """
          Args:
            net: (object) net & optimizer
            datasets : (list) [train, val] dataset object
        """
        args = self.args
        best_trn_perform, best_val_perform = -1, -1
        steps = len(datasets[0]) // args.batch_size
        if steps * args.batch_size < len(datasets[0]):
            steps += 1

        if args.trigger == 'epoch':
            args.epochs = args.terminal
            args.iters = steps * args.terminal
            args.iter_interval = steps * args.interval
        else:
            args.epochs = args.terminal // steps + 1
            args.iters = args.terminal
            args.iter_interval = args.interval

        net.train()
        trn_loss, trn_acc = [], []
        start = time.time()
        criterionA = losses.DSSIMLoss3D()
        criterionB = losses.L1Loss3D()
        criterionExtra = losses.VGG16Loss3D()
        for epoch in range(1, args.epochs + 1):
            self.epoch = epoch
            # setup data loader
            data_loader = DataLoader(datasets[0], args.batch_size, num_workers=4,
                                     shuffle=True, pin_memory=True,)
            for idx, sample in enumerate(data_loader):
                self.iter += 1
                if self.iter > args.iters:
                    self.iter -= 1
                    break
                # get tensors from sample
                x = sample["src"]
                y = sample["tar"]
                if args.cuda:
                    x = x.cuda()
                    y = y.cuda()
                # forwading
                # x => [batch, ch, frame, row, col]
                gen_l, gen_ab = net(x)
                gen_y = torch.cat([gen_l, gen_ab], dim=1)
                # loss_l = self.criterion(gen_y[:,:1,:,:], y[:,:1,:,:])
                # loss_ab = self.criterion(gen_y[:,1:,:,:], y[:,1:,:,:])
                loss_l = criterionA(gen_y[:,:1,:,:,:], y[:,:1,:,:,:])
                loss_ab = criterionB(gen_y[:,1:,:,:,:], y[:,1:,:,:,:])
                loss_lab = criterionExtra(gen_y, y)
                loss = loss_l + loss_ab + args.alpha * loss_lab
                # update parameters
                net.optimizer.zero_grad()
                loss.backward()
                net.optimizer.step()
                # update taining condition
                trn_loss.append(loss.detach().item())
                acc = []
                for f in range(y.shape[2]):
                    acc.append(self.evaluator(gen_y[:,:,f,:,:].detach(), y[:,:,f,:,:].detach()).item())
                trn_acc.append(sum(acc)/len(acc))
                # validating
                if self.iter % args.iter_interval == 0:
                    trn_fps = (args.iter_interval * args.batch_size) / (time.time() - start)
                    self.trn_log = [round(sum(trn_loss) / len(trn_loss), 3), 
                                    round(sum(trn_acc) / len(trn_acc), 3),
                                    round(trn_fps, 3)]
 
                    self.validating(net, datasets[1])
                    self.logging(verbose=True)
                    if self.val_log[1] >= best_val_perform:
                        best_trn_perform = self.trn_log[1]
                        best_val_perform = self.val_log[1]
                        checkpoint_info = [self.repr, self.epoch, self.iter,
                                           best_trn_perform, best_val_perform]
                        # save better checkpoint
                        self.save_checkpoint(net)
                    # reinitialize
                    start = time.time()
                    trn_loss, trn_acc = [], []
                    net.train()

        df = pd.DataFrame([checkpoint_info],
                          columns=["experiment", "best_epoch", "best_iter", self.header[3], self.header[6]])
        if os.path.exists(os.path.join(Checkpoint_DIR, 'checkpoint.csv')):
            prev_df = pd.read_csv(os.path.join(Checkpoint_DIR, 'checkpoint.csv'))
            df = prev_df.append(df)
        df.to_csv(os.path.join(Checkpoint_DIR, 'checkpoint.csv'), index=False,  float_format='%.3f')

        print("Best {} Performance: \n".format(repr(self.evaluator)))
        print("\t Trn:", best_trn_perform)
        print("\t Val:", best_val_perform)

    def validating(self, net, dataset):
        """
          Args:
            net: (object) pytorch net
            batch_size: (int)
            dataset : (object) dataset
          return [loss, acc]
        """
        args = self.args
        data_loader = DataLoader(dataset, args.batch_size, num_workers=4,
                                 shuffle=False, pin_memory=True,)
        val_loss, val_acc = [], []
        start = time.time()
        net.eval()
        with torch.set_grad_enabled(False):
            for idx, sample in enumerate(data_loader):
                # get tensors from sample
                x = sample["src"]
                y = sample["tar"]
                if args.cuda:
                    x = x.cuda()
                    y = y.cuda()
                # forwading
                gen_l, gen_ab = net(x)
                gen_y = torch.cat([gen_l, gen_ab], dim=1)
                loss, acc = [], []
                for f in range(y.shape[2]):
                    loss.append(self.criterion(gen_y[:,:,f,:,:].detach(), y[:,:,f,:,:].detach()).item())
                    acc.append(self.evaluator(gen_y[:,:,f,:,:].detach(), y[:,:,f,:,:].detach()).item())
                val_loss.append(sum(loss)/len(loss))
                val_acc.append(sum(acc)/len(acc))

        val_fps = (len(val_loss) * args.batch_size) / (time.time() - start)
        self.val_log = [round(sum(val_loss) / len(val_loss), 3), 
                        round(sum(val_acc) / len(val_acc), 3),
                        round(val_fps, 3)]
        self.save_snapshot(x.detach(), y.detach(), gen_y.detach(), dataset)

class bergTrainer(Trainer):
    def training(self, net, datasets):
        """
          Args:
            net: (object) net & optimizer
            datasets : (list) [train, val] dataset object
        """
        args = self.args
        best_trn_perform, best_val_perform = -1, -1
        steps = len(datasets[0]) // args.batch_size
        if steps * args.batch_size < len(datasets[0]):
            steps += 1

        if args.trigger == 'epoch':
            args.epochs = args.terminal
            args.iters = steps * args.terminal
            args.iter_interval = steps * args.interval
        else:
            args.epochs = args.terminal // steps + 1
            args.iters = args.terminal
            args.iter_interval = args.interval

        net.train()
        trn_loss, trn_acc = [], []
        start = time.time()
        criterionA = losses.DSSIMLoss()
        criterionB = losses.L1Loss()
        for epoch in range(1, args.epochs + 1):
            self.epoch = epoch
            # setup data loader
            data_loader = DataLoader(datasets[0], args.batch_size, num_workers=4,
                                     shuffle=True, pin_memory=True,)
            for idx, sample in enumerate(data_loader):
                self.iter += 1
                if self.iter > args.iters:
                    self.iter -= 1
                    break
                # get tensors from sample
                x = sample["src"]
                y = sample["tar"]
                if args.cuda:
                    x = x.cuda()
                    y = y.cuda()
                # forwading
                gen_y = net(x)
                # loss = criterionB(gen_y, y)
                loss_l = criterionA(gen_y[:,:1,:,:], y[:,:1,:,:])
                loss_ab = criterionB(gen_y[:,1:,:,:], y[:,1:,:,:])
                loss = loss_l + loss_ab
                # update parameters
                net.optimizer.zero_grad()
                loss.backward()
                net.optimizer.step()
                # update taining condition
                trn_loss.append(loss.detach().item())
                trn_acc.append(self.evaluator(gen_y.detach(), y.detach()).item())
                # validating
                if self.iter % args.iter_interval == 0:
                    trn_fps = (args.iter_interval * args.batch_size) / (time.time() - start)
                    self.trn_log = [round(sum(trn_loss) / len(trn_loss), 3), 
                                    round(sum(trn_acc) / len(trn_acc), 3),
                                    round(trn_fps, 3)]
 
                    self.validating(net, datasets[1])
                    self.logging(verbose=True)
                    if self.val_log[1] >= best_val_perform:
                        best_trn_perform = self.trn_log[1]
                        best_val_perform = self.val_log[1]
                        checkpoint_info = [self.repr, self.epoch, self.iter,
                                           best_trn_perform, best_val_perform]
                        # save better checkpoint
                        self.save_checkpoint(net)
                    # reinitialize
                    start = time.time()
                    trn_loss, trn_acc = [], []
                    net.train()

        df = pd.DataFrame([checkpoint_info],
                          columns=["experiment", "best_epoch", "best_iter", self.header[3], self.header[6]])
        if os.path.exists(os.path.join(Checkpoint_DIR, 'checkpoint.csv')):
            prev_df = pd.read_csv(os.path.join(Checkpoint_DIR, 'checkpoint.csv'))
            df = prev_df.append(df)
        df.to_csv(os.path.join(Checkpoint_DIR, 'checkpoint.csv'), index=False,  float_format='%.3f')

        print("Best {} Performance: \n".format(repr(self.evaluator)))
        print("\t Trn:", best_trn_perform)
        print("\t Val:", best_val_perform)

    def validating(self, net, dataset):
        """
          Args:
            net: (object) pytorch net
            batch_size: (int)
            dataset : (object) dataset
          return [loss, acc]
        """
        args = self.args
        data_loader = DataLoader(dataset, args.batch_size, num_workers=4,
                                 shuffle=False, pin_memory=True,)
        val_loss, val_acc = [], []
        start = time.time()
        net.eval()
        criterionA = losses.DSSIMLoss()
        criterionB = losses.L1Loss()
        with torch.set_grad_enabled(False):
            for idx, sample in enumerate(data_loader):
                # get tensors from sample
                x = sample["src"]
                y = sample["tar"]
                if args.cuda:
                    x = x.cuda()
                    y = y.cuda()
                # forwading
                gen_y = net(x)
                loss_l = criterionA(gen_y[:,:1,:,:], y[:,:1,:,:])
                loss_ab = criterionB(gen_y[:,1:,:,:], y[:,1:,:,:])
                loss = loss_l + loss_ab
                val_loss.append(loss.detach().item())
                val_acc.append(self.evaluator(gen_y.detach(), y.detach()).item())

        val_fps = (len(val_loss) * args.batch_size) / (time.time() - start)
        self.val_log = [round(sum(val_loss) / len(val_loss), 3), 
                        round(sum(val_acc) / len(val_acc), 3),
                        round(val_fps, 3)]
        self.save_snapshot(x.detach(), y.detach(), gen_y.detach(), dataset)

