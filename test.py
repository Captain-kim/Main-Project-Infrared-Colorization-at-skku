#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
  @Email:  guangmingwu2010@gmail.com
  @Copyright: go-hiroaki
  @License: MIT
"""
import argparse
import os
import time
import torch
import numpy as np
import pandas as pd

from datasets import load_dataset
from utils import load_checkpoint, load_checkpoint_custom
from skimage.io import imread, imsave
from torch.utils.data import DataLoader


DIR = os.path.dirname(os.path.abspath(__file__))
Result_DIR = os.path.join(DIR, 'result')
if not os.path.exists(os.path.join(Result_DIR, 'frame')):
    os.mkdir(os.path.join(Result_DIR, 'frame'))


def saving_ref(root, ver='RGB2RGB'):
    src_dir = os.path.join(Result_DIR, 'frame', '{}-src'.format(root))
    tar_dir = os.path.join(Result_DIR, 'frame', '{}-tar'.format(root))
    _, _, testset = load_dataset(root, ver, "evaluate")
    if not os.path.exists(src_dir):
        os.mkdir(src_dir)
        os.mkdir(tar_dir)
        data_loader = DataLoader(testset, 1, num_workers=4,
                                 shuffle=False, pin_memory=True,)
        for idx, sample in enumerate(data_loader):
            # get tensors from sample
            x = sample["src"]
            y = sample["tar"]
            src_img = testset._rgb2img(x.numpy()[0].transpose((1, 2, 0)), False)
            tar_img = testset._rgb2img(y.numpy()[0].transpose((1, 2, 0)), False)
            filename = testset.datalist[idx]
            imsave(os.path.join(src_dir, filename), src_img)
            imsave(os.path.join(tar_dir, filename), tar_img)
    return 0


def main(args):
    if args.cuda and not torch.cuda.is_available():
        raise ValueError("GPUs are not available, please run at cpu mode")

    for checkpoint in args.checkpoints:
        Save_DIR = os.path.join(Result_DIR, 'frame', checkpoint.split("_")[0])
        if not os.path.exists(Save_DIR):
            os.makedirs(Save_DIR)
        # initialize datasets
        infos = checkpoint.split('_')[0].split('-')
        # saving_ref(infos[1], ver='RGB2RGB')
        _, _, testset = load_dataset(args.root, infos[2], "evaluate")
        print("Testing with {}/{}-Dataset: {} examples".format(args.root, infos[2], len(testset)))
        args.src_ch = testset.src_ch
        args.tar_ch = testset.tar_ch
        # Load checkpoint
        net1, net2, net3 = load_checkpoint_custom(checkpoint, testset.src_ch, testset.tar_ch, args.cuda)

        # load data
        data_loader = DataLoader(testset, args.batch_size, num_workers=4,
                                 shuffle=False, pin_memory=True,)
        count = 0
        net1.eval()
        net2.eval()
        net3.eval()
        with torch.set_grad_enabled(False):
            for idx, sample in enumerate(data_loader):
                # get tensors from sample
                x = sample["src"]
                y = sample["tar"]
                fsets = sample["idx"]
                if args.cuda:
                    x = x.cuda()
                    y = y.cuda()
                # forwading

                if 'ynet' in infos[0]:
                    gen_l, gen_ab = net(x)
                    gen_y = torch.cat([gen_l.detach(), gen_ab.detach()], dim=1)
                elif 'attention' in infos[0]:
                    gen_R, gen_G, gen_B = net(x)
                    gen_y = torch.cat([gen_R.detach(), gen_G.detach(), gen_B.detach()], dim=1)
                elif 'Lab' in infos[0]:
                    gen_l, gen_a, gen_b = net(x)
                    gen_y = torch.cat([gen_l.detach(), gen_a.detach(), gen_b.detach()], dim=1)
                elif 'VAE' in infos[0]:
                    gen_l, gen_a, gen_b, _, _ = net(x)
                    gen_y = torch.cat([gen_l.detach(), gen_a.detach(), gen_b.detach()], dim=1)
                elif 'VQ' in infos[0]:
                    gen_l, gen_a, gen_b, _, _ = net(x)
                    gen_y = torch.cat([gen_l.detach(), gen_a.detach(), gen_b.detach()], dim=1)
                elif 'WGAN' in infos[0]:
                    gen_l, gen_a, gen_b, _, _ = net(x)
                    gen_y = torch.cat([gen_l.detach(), gen_a.detach(), gen_b.detach()], dim=1)
                elif 'pix2pixHD' in infos[0]:
                    gen_l, gen_a, gen_b, _, _ = net(x)
                    gen_y = torch.cat([gen_l.detach(), gen_a.detach(), gen_b.detach()], dim=1)
                elif 'custom' in infos[0]:
                    gen_l_i, gen_a_i, gen_b_i, _, _ = net1(x)
                    gen_l_e, gen_a_e, gen_b_e, _, _ = net2(x)
                    gen_i = torch.cat([gen_l_i, gen_a_i, gen_b_i], dim=1)
                    gen_e = torch.cat([gen_l_e, gen_a_e, gen_b_e], dim=1)
                    gen_input = torch.cat([gen_i, gen_e], dim=1)
                    gen_l, gen_a, gen_b, _, _ = net3(gen_input)
                    gen_y = torch.cat([gen_l.detach(), gen_a.detach(), gen_b.detach()], dim=1)
                elif 'pix2pixWGAN' in infos[0]:
                    gen_l, gen_a, gen_b, _, _ = net(x)
                    gen_y = torch.cat([gen_l.detach(), gen_a.detach(), gen_b.detach()], dim=1)
                elif 'final' in infos[0]:
                    _, _, _, x10_L_G, x10_a_G, x10_b_G = net1(x)
                    _, _, _, x10_L_e, x10_a_e, x10_b_e = net2(x)
                    x10_G = torch.cat([x10_L_G.detach(), x10_a_G.detach(), x10_b_G.detach()], dim=1)
                    x10_e = torch.cat([x10_L_e.detach(), x10_a_e.detach(), x10_b_e.detach()], dim=1)
                    x10_input = torch.cat([x10_G, x10_e], dim=1)
                    gen_l, gen_a, gen_b = net3(x10_input)
                    gen_y = torch.cat([gen_l.detach(), gen_a.detach(), gen_b.detach()], dim=1)
                elif 'beta' in infos[0]:
                    gen_L_G, gen_ab_G = net1(x)
                    gen_L_e, gen_ab_e = net2(x)
                    gen_G = torch.cat([gen_L_G.detach(), gen_ab_G.detach()], dim=1)
                    gen_e = torch.cat([gen_L_e.detach(), gen_ab_e.detach()], dim=1)
                    #gen_input = torch.cat([gen_G.detach(), gen_e.detach()], dim=1)
                    gen_l, gen_ab = net3(gen_G, gen_e)
                    gen_y = torch.cat([gen_l.detach(), gen_ab.detach()], dim=1)
                else:
                    gen_y = net(x).detach()

                # save prediction
                if args.cuda:
                    x = x.cpu().numpy()
                    y = y.cpu().numpy()
                    gen_y = gen_y.cpu().numpy()

                if '3D' in infos[2]:
                    for idxi, fset in enumerate(fsets):
                        fset = testset.datalist[int(fset)]
                        for idxj, filename in enumerate(fset):
                            count += 1
                            print('\t Processing  {} {:03d} / {:03d} '.format(filename, count, len(testset)*len(fset)),
                                end='\r', flush=True)
                            src = x[idxi,:,idxj].transpose((1, 2, 0))
                            tar = y[idxi,:,idxj].transpose((1, 2, 0))
                            gen = gen_y[idxi,:,idxj].transpose((1, 2, 0))
                            src_type, tar_type = infos[2].split("2")[:2]

                            if "RGB" in tar_type:
                                gen_img = testset._rgb2img(gen, False)
                            else:
                                gen_img = testset._lab2img(gen, False)
                            # save image
                            imsave(os.path.join(Save_DIR, filename), gen_img)
                else:
                    for idx, fid in enumerate(fsets):
                        count += 1
                        filename = testset.datalist[int(fid)]
                        print('\t Processing  {} {:03d} / {:03d} '.format(filename, count, len(testset)),
                            end='\r', flush=True)
                        src = x[idx].transpose((1, 2, 0))
                        tar = y[idx].transpose((1, 2, 0))
                        gen = gen_y[idx].transpose((1, 2, 0))
                        src_type, tar_type = infos[2].split("2")[:2]

                        if "LAB" != tar_type:
                            gen_img = testset._rgb2img(gen, False)
                        else:
                            gen_img = testset._lab2img(gen, False)
                        # save image
                        imsave(os.path.join(Save_DIR, filename), gen_img)

if __name__ == "__main__":
    # ====================== parameter initialization ======================= #
    parser = argparse.ArgumentParser(description='ArgumentParser')
    parser.add_argument('-checkpoints', nargs='+', type=str, default=["res18unet-S6A-RGB2RGB_iter_10000.pth"],
                        help='checkpoints used for making prediction ')
    parser.add_argument('-root', type=str, default='VC24',
                        help='root dir of dataset ')
    parser.add_argument('-batch_size', type=int, default=5,
                        help='batch_size for training ')
    parser.add_argument('-cuda', type=lambda x: (str(x).lower() == 'true'), default=True,
                        help='using cuda for optimization')
    args = parser.parse_args()

    main(args)