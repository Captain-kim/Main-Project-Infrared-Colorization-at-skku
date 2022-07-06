#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
  @Email:  guangmingwu2010@gmail.com
  @Copyright: go-hiroaki
  @License: MIT
"""
import os
import re
import torch
import torch.nn as nn
from models.fcn import fcn8s, fcn16s, fcn32s
from models.unet import unet
from models.fpn import fpn
from models.resnet import res18net, res34net
from models.resunet import res18unet, res34unet
from models.resunetnb import res18unetNB, res34unetNB
from models.resynet import res18ynet, res34ynet
from models.resynetsync import res18ynetsync, res34ynetsync, resnetpix2pixbeta7
from models.resynetsyncnb import res18ynetsyncNB, res34ynetsyncNB
from models.my_attention_resnet import attentionres18net6, attentionres34net, res18netLab, resnetVAE3, resnetVAE34, resnetVQ2, resnetVQ, resnetVQ3, resnetWGAN
from models.my_attention_resnet import resnetpix2pixHD, resnetpix2pixHD2, resnetpix2pixHD3, resnetpix2pixHD4, resnetpix2pixWGAN, resnetpix2pixWGAN2, resnetpix2pixWGAN3
from models.my_attention_resnet import resnetpix2pixWGAN4, resnetpix2pixcustom, resnetpix2pixcustom3
from models.my_attention_resnet import resnetpix2pixfinal, resnetpix2pixfinal6

from torch.nn import Parameter

DIR = os.path.dirname(os.path.abspath(__file__))
Checkpoint_DIR = os.path.join(DIR, '../checkpoint')

def l2normalize(v, eps = 1e-12):
    return v / (v.norm() + eps)

class SpectralNorm(nn.Module):
    def __init__(self, module, name = 'weight', power_iterations = 1):
        super(SpectralNorm, self).__init__()
        self.module = module
        self.name = name
        self.power_iterations = power_iterations
        if not self._made_params():
            self._make_params()

    def _update_u_v(self):
        u = getattr(self.module, self.name + "_u")
        v = getattr(self.module, self.name + "_v")
        w = getattr(self.module, self.name + "_bar")

        height = w.data.shape[0]
        for _ in range(self.power_iterations):
            v.data = l2normalize(torch.mv(torch.t(w.view(height, -1).data), u.data))
            u.data = l2normalize(torch.mv(w.view(height, -1).data, v.data))

        # sigma = torch.dot(u.data, torch.mv(w.view(height,-1).data, v.data))
        sigma = u.dot(w.view(height, -1).mv(v))
        setattr(self.module, self.name, w / sigma.expand_as(w))

    def _made_params(self):
        try:
            u = getattr(self.module, self.name + "_u")
            v = getattr(self.module, self.name + "_v")
            w = getattr(self.module, self.name + "_bar")
            return True
        except AttributeError:
            return False

    def _make_params(self):
        w = getattr(self.module, self.name)

        height = w.data.shape[0]
        width = w.view(height, -1).data.shape[1]

        u = Parameter(w.data.new(height).normal_(0, 1), requires_grad = False)
        v = Parameter(w.data.new(width).normal_(0, 1), requires_grad = False)
        u.data = l2normalize(u.data)
        v.data = l2normalize(v.data)
        w_bar = Parameter(w.data)

        del self.module._parameters[self.name]

        self.module.register_parameter(self.name + "_u", u)
        self.module.register_parameter(self.name + "_v", v)
        self.module.register_parameter(self.name + "_bar", w_bar)

    def forward(self, *args):
        self._update_u_v()
        return self.module.forward(*args)


class Conv2dLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, pad_type='zero',
                 activation='lrelu', norm='none', sn=True):
        super(Conv2dLayer, self).__init__()
        # Initialize the padding scheme
        if pad_type == 'reflect':
            self.pad = nn.ReflectionPad2d(padding)
        elif pad_type == 'replicate':
            self.pad = nn.ReplicationPad2d(padding)
        elif pad_type == 'zero':
            self.pad = nn.ZeroPad2d(padding)
        else:
            assert 0, "Unsupported padding type: {}".format(pad_type)

        # Initialize the normalization type
        if norm == 'bn':
            self.norm = nn.BatchNorm2d(out_channels)
        elif norm == 'in':
            self.norm = nn.InstanceNorm2d(out_channels)
        elif norm == 'ln':
            self.norm = LayerNorm(out_channels)
        elif norm == 'none':
            self.norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(norm)

        # Initialize the activation funtion
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == 'prelu':
            self.activation = nn.PReLU()
        elif activation == 'selu':
            self.activation = nn.SELU(inplace=True)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif activation == 'none':
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(activation)

        # Initialize the convolution layers
        if sn:
            self.conv2d = SpectralNorm(
                nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=0, dilation=dilation))
        else:
            self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=0, dilation=dilation)

    def forward(self, x):
        x = self.pad(x)
        x = self.conv2d(x)
        if self.norm:
            x = self.norm(x)
        if self.activation:
            x = self.activation(x)
        return x

# This is a kind of PatchGAN. Patch is implied in the output. This is 70 * 70 PatchGAN
class PatchDiscriminator70(nn.Module):
    def __init__(self):
        super(PatchDiscriminator70, self).__init__()
        # Down sampling
        self.block1 = Conv2dLayer(3, 64, 7, 1, 3, pad_type = 'reflect', activation = 'lrelu', norm = 'bn', sn = True)
        self.block2 = Conv2dLayer(64, 64 * 2, 3, 2, 1, pad_type = 'reflect', activation = 'lrelu', norm = 'bn', sn = True)
        self.block3 = Conv2dLayer(64 * 2, 64 * 4, 3, 2, 1, pad_type = 'reflect', activation = 'lrelu', norm = 'bn', sn = True)
        self.block4 = Conv2dLayer(64 * 4, 64 * 8, 3, 2, 1, pad_type = 'reflect', activation = 'lrelu', norm = 'bn', sn = True)
        # Final output, implemention of 70 * 70 PatchGAN
        self.final1 = Conv2dLayer(64 * 8, 64 * 8, 4, 1, 1, pad_type = 'reflect', activation = 'lrelu', norm = 'bn', sn = True)
        self.final2 = Conv2dLayer(64 * 8, 1, 4, 1, 1, pad_type = 'reflect', activation = 'none', norm = 'none', sn = True)

    def forward(self, x):
        # img_A: input grayscale image
        # img_B: generated color image or ground truth color image; generated weighted image or ground truth weighted image
        # Concatenate image and condition image by channels to produce input
        #x = torch.cat((img_A, img_B), 1)                        # out: batch * 4 * 256 * 256
        # Inference
        x = self.block1(x)                                      # out: batch * 64 * 256 * 256
        x = self.block2(x)                                      # out: batch * 128 * 128 * 128
        x = self.block3(x)                                      # out: batch * 256 * 64 * 64
        x = self.block4(x)                                      # out: batch * 512 * 32 * 32
        x = self.final1(x)                                      # out: batch * 512 * 31 * 31
        x = self.final2(x)                                      # out: batch * 1 * 30 * 30
        return x
def weights_init(net, init_type = 'normal', init_gain = 0.02):
    """Initialize network weights.
    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal
    In our paper, we choose the default setting: zero mean Gaussian distribution with a standard deviation of 0.02
    """
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and classname.find('Conv') != -1:
            if init_type == 'normal':
                torch.nn.init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                torch.nn.init.xavier_normal_(m.weight.data, gain = init_gain)
            elif init_type == 'kaiming':
                torch.nn.init.kaiming_normal_(m.weight.data, a = 0, mode = 'fan_in')
            elif init_type == 'orthogonal':
                torch.nn.init.orthogonal_(m.weight.data, gain = init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
        elif classname.find('BatchNorm2d') != -1:
            torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
            torch.nn.init.constant_(m.bias.data, 0.0)

    # apply the initialization function <init_func>
    print('initialize network with %s type' % init_type)
    net.apply(init_func)

# ----------------------------------------

def create_discriminator():
    # Initialize the networks
    discriminator = PatchDiscriminator70()
    # Init the networks
    weights_init(discriminator, init_type = 'xavier', init_gain = 0.02)
    return discriminator


def load_model(net, src_ch, tar_ch, cuda):
    if net == "res18ynetsyncGN":
        net = eval('res18ynetsync')(src_ch, tar_ch, True, 'GN')
    else:
        net = eval(net)(src_ch, tar_ch, True, 'IN')
    if cuda:
        net.cuda()
    return net

def load_model_custom(net, src_ch, tar_ch, cuda):
    if net == "res18ynetsyncGN":
        net = eval('res18ynetsync')(src_ch, tar_ch, True, 'GN')
    else:
        net = eval(net)(src_ch, tar_ch, True, 'IN')
    if cuda:
        net[0].cuda()
        net[1].cuda()
        net[2].cuda()
    return net

def load_checkpoint(checkpoint, src_ch, tar_ch, cuda):
    assert os.path.exists("{}/{}".format(Checkpoint_DIR, checkpoint)
                          ), "{} not exists.".format(checkpoint)
    print("Loading checkpoint: {}".format(checkpoint))
    net = checkpoint.split('-')[0]
    if "@" in net:
        net = net.split('@')[0]
    if net == "res18ynetsyncGN":
        net = eval('res18ynetsync')(src_ch, tar_ch, False, 'GN')
    else:
        net = eval(net)(src_ch, tar_ch, False, 'IN')
    net.load_state_dict(torch.load(os.path.join(Checkpoint_DIR, checkpoint),
                                   map_location=lambda storage, loc: storage))
    if cuda:
        net.cuda()
    return net.eval()

def load_checkpoint_custom(checkpoint, src_ch, tar_ch, cuda):
    print("Loading checkpoint: {}".format(checkpoint))
    net = checkpoint.split('-')[0]

    if "@" in net:
        net = net.split('@')[0]
    if net == "res18ynetsyncGN":
        net = eval('res18ynetsync')(src_ch, tar_ch, False, 'GN')
    else:
        net = eval(net)(src_ch, tar_ch, False, 'IN')
    net[0].load_state_dict(torch.load(os.path.join(Checkpoint_DIR, "0" + checkpoint),
                                      map_location=lambda storage, loc: storage))
    net[1].load_state_dict(torch.load(os.path.join(Checkpoint_DIR, "1" + checkpoint),
                                      map_location=lambda storage, loc: storage))
    net[2].load_state_dict(torch.load(os.path.join(Checkpoint_DIR, "2" + checkpoint),
                                      map_location=lambda storage, loc: storage))
    if cuda:
        net[0].cuda()
        net[1].cuda()
        net[2].cuda()
    return net[0].eval(), net[1].eval(), net[2].eval()

def natural_sort(unsorted_list):
    # refer to  https://stackoverflow.com/questions/4836710/does-python-have-a-built-in-function-for-string-natural-sort
    def convert(text): return int(text) if text.isdigit() else text

    def alphanum_key(key): return [convert(c)
                                   for c in re.split('([0-9]+)', key)]
    return sorted(unsorted_list, key=alphanum_key)


def pair_validate(pFiles, gFiles):
    valid = True
    if len(pFiles) == len(gFiles):
        for pfile, gfile in zip(pFiles, gFiles):
            if os.path.basename(pfile) != os.path.basename(gfile):
                valid = False
                # print('{} and {} isn\'t consistent.'.format(os.path.basename(pfile), os.path.basename(gfile)))
    else:
        valid = False
        # find different
        xlen = min(len(pFiles), len(gFiles))
        for pfile, gfile in zip(pFiles[:xlen], gFiles[:xlen]):
            if os.path.basename(pfile) != os.path.basename(gfile):
                print('{} and {} isn\'t consistent.'.format(os.path.basename(pfile), os.path.basename(gfile)))
        print("Warning >> Extra P:\n", len(pFiles[xlen:]))
        print("Warning >> Extra G:\n", len(gFiles[xlen:]))
    return valid
