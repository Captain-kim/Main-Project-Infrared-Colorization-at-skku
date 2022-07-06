import argparse
import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from torch.autograd import Variable
from .block import *
import distributed as dist_fn
import cv2

__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152']

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


class adaILN(nn.Module):
    def __init__(self, num_features, eps=1e-5):
        super(adaILN, self).__init__()
        self.eps = eps
        self.rho = nn.Parameter(torch.Tensor(1, num_features, 1, 1))
        self.rho.data.fill_(0.9)

    def forward(self, input, gamma, beta):
        in_mean, in_var = torch.mean(input, dim=[2, 3], keepdim=True), torch.var(input, dim=[2, 3], keepdim=True)
        out_in = (input - in_mean) / torch.sqrt(in_var + self.eps)
        ln_mean, ln_var = torch.mean(input, dim=[1, 2, 3], keepdim=True), torch.var(input, dim=[1, 2, 3], keepdim=True)
        out_ln = (input - ln_mean) / torch.sqrt(ln_var + self.eps)
        out = self.rho.expand(input.shape[0], -1, -1, -1) * out_in \
              + (1 - self.rho.expand(input.shape[0], -1, -1, -1)) * out_ln
        out = out * gamma.unsqueeze(2).unsqueeze(3) + beta.unsqueeze(2).unsqueeze(3)

        return out


class Interp(nn.Module):
    def __init__(self, scale_factor=2, mode='nearest', align_corners=None, utype='2D'):
        super(Interp, self).__init__()
        self.up = F.interpolate
        self.mode = mode
        self.align_corners = align_corners
        if utype == '2D':
            self.scale_factors = [scale_factor, scale_factor]
        elif utype == '3D':
            self.scale_factors = [1] + [scale_factor, scale_factor]
        else:
            raise ValueError('{} is not support'.format(utype))

    def forward(self, x):
        x = self.up(x, scale_factor=tuple(self.scale_factors), mode=self.mode, align_corners=self.align_corners)
        return x


def upconv(in_planes, out_planes, ratio="x2"):
    """2d upsampling"""
    return nn.Sequential(
        Interp(scale_factor=2, mode='nearest', align_corners=None, utype='2D'),
        conv1x1(in_planes, out_planes)
    )


class AFF(nn.Module):

    def __init__(self, channels=64, r=4):
        super(AFF, self).__init__()
        inter_channels = int(channels // r)

        self.local_att = nn.Sequential(
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x, residual):
        xa = x + residual
        xl = self.local_att(xa)
        xg = self.global_att(xa)
        xlg = xl + xg
        wei = self.sigmoid(xlg)

        xo = 2 * x * wei + 2 * residual * (1 - wei)
        return xo


class Attention_block(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, BNmode='BN', fuse_type='AFF'):
        super(Attention_block, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        if BNmode == 'BN':
            self.bn1 = nn.BatchNorm2d(planes)
        elif BNmode == 'IN':
            self.bn1 = nn.InstanceNorm2d(planes)
        elif BNmode == 'GN':
            self.bn1 = nn.GroupNorm(32, planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        if BNmode == 'BN':
            self.bn2 = nn.BatchNorm2d(planes)
        elif BNmode == 'IN':
            self.bn2 = nn.InstanceNorm2d(planes)
        elif BNmode == 'GN':
            self.bn2 = nn.GroupNorm(32, planes)
        self.downsample = downsample
        self.stride = stride

        if fuse_type == 'AFF':
            self.fuse_mode = AFF(channels=planes)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.fuse_mode(out, identity)
        out = self.relu(out)

        return out


class block(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, BNmode='BN'):
        super(block, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        if BNmode == 'BN':
            self.bn1 = nn.BatchNorm2d(planes)
        elif BNmode == 'IN':
            self.bn1 = nn.InstanceNorm2d(planes)
        elif BNmode == 'GN':
            self.bn1 = nn.GroupNorm(32, planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        if BNmode == 'BN':
            self.bn2 = nn.BatchNorm2d(planes)
        elif BNmode == 'IN':
            self.bn2 = nn.InstanceNorm2d(planes)
        elif BNmode == 'GN':
            self.bn2 = nn.GroupNorm(32, planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)

        return out


class attentionResnetSync(nn.Module):

    def __init__(self, block, Attention_block, layers, src_ch, tar_ch, BNmode):
        super(attentionResnetSync, self).__init__()
        kernels = [64, 128, 256, 512]
        self.epsilon = 1e-4
        self.alpha = nn.Parameter(torch.ones(3))
        self.weight_act = nn.ReLU()
        self.src_ch = src_ch
        self.inplanes = kernels[0]
        self.stem_conv = nn.Conv2d(3, kernels[0], kernel_size=7,
                                   stride=2, padding=3, bias=False)
        print("####1 BNmode => ", BNmode)
        if BNmode == 'BN':
            self.stem_norm = nn.BatchNorm2d(kernels[0])
        elif BNmode == 'IN':
            self.stem_norm = nn.InstanceNorm2d(kernels[0])
        elif BNmode == 'GN':
            self.stem_norm = nn.GroupNorm(32, kernels[0])
        # Adding AdaIN, combined BN and IN
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(block, kernels[0], layers[0], stride=1, BNmode=BNmode)
        self.layer2 = self._make_layer(block, kernels[1], layers[1], stride=2, BNmode=BNmode)
        self.layer3 = self._make_layer(Attention_block, kernels[2], layers[2], stride=2, BNmode=BNmode)
        self.layer4 = self._make_layer(Attention_block, kernels[3], layers[3], stride=2, BNmode=BNmode)

        # Red components
        self.deconvR1 = upconv(kernels[3], kernels[2], ratio="x2")
        self.inplanes = kernels[2]
        self.upResR1 = self._make_layer(block, kernels[2], layers[2], BNmode=BNmode)
        self.deconvR2 = upconv(kernels[2], kernels[1], ratio="x2")
        self.inplanes = kernels[1]
        self.upResR2 = self._make_layer(block, kernels[1], layers[1], BNmode=BNmode)
        self.deconvR3 = upconv(kernels[1], kernels[0], ratio="x2")
        self.inplanes = kernels[0]
        self.upResR3 = self._make_layer(block, kernels[0], layers[0], BNmode=BNmode)
        self.deconvR4 = upconv(kernels[0], kernels[0], ratio="x2")
        self.predR = nn.Conv2d(kernels[0], tar_ch[0], kernel_size=3, stride=1, padding=1, bias=False)

        # Green components
        self.deconvG1 = upconv(kernels[3], kernels[2], ratio="x2")
        self.inplanes = kernels[2]
        self.upResG1 = self._make_layer(block, kernels[2], layers[2], BNmode=BNmode)
        self.deconvG2 = upconv(kernels[2], kernels[1], ratio="x2")
        self.inplanes = kernels[1]
        self.upResG2 = self._make_layer(block, kernels[1], layers[1], BNmode=BNmode)
        self.deconvG3 = upconv(kernels[1], kernels[0], ratio="x2")
        self.inplanes = kernels[0]
        self.upResG3 = self._make_layer(block, kernels[0], layers[0], BNmode=BNmode)
        self.deconvG4 = upconv(kernels[0], kernels[0], ratio="x2")
        self.predG = nn.Conv2d(kernels[0], tar_ch[1], kernel_size=3, stride=1, padding=1, bias=False)

        # Blue components
        self.deconvB1 = upconv(kernels[3], kernels[2], ratio="x2")
        self.inplanes = kernels[2]
        self.upResB1 = self._make_layer(block, kernels[2], layers[2], BNmode=BNmode)
        self.deconvB2 = upconv(kernels[2], kernels[1], ratio="x2")
        self.inplanes = kernels[1]
        self.upResB2 = self._make_layer(block, kernels[1], layers[1], BNmode=BNmode)
        self.deconvB3 = upconv(kernels[1], kernels[0], ratio="x2")
        self.inplanes = kernels[0]
        self.upResB3 = self._make_layer(block, kernels[0], layers[0], BNmode=BNmode)
        self.deconvB4 = upconv(kernels[0], kernels[0], ratio="x2")
        self.predB = nn.Conv2d(kernels[0], tar_ch[2], kernel_size=3, stride=1, padding=1, bias=False)

        self.final_conv_1x1 = conv1x1(in_planes=3, out_planes=3)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1, BNmode='BN'):
        downsample = None
        print("####2 BNmode => ", BNmode)
        if stride != 1 or self.inplanes != planes * block.expansion:
            bnl = nn.BatchNorm2d(planes * block.expansion)
            if BNmode == 'IN':
                bnl = nn.InstanceNorm2d(planes * block.expansion)
            if BNmode == 'GN':
                bnl = nn.GroupNorm(32, planes * block.expansion)
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                bnl,
            )
            # Adding AdaIN

        layers = []
        if BNmode == 'GN':
            layers.append(block(self.inplanes, planes, stride, downsample, BNmode=BNmode))
        else:
            layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            if BNmode == 'GN':
                layers.append(block(self.inplanes, planes, BNmode=BNmode))
            else:
                layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    @staticmethod
    def _gaussian_filter(channel, kernel_size, sigma=1.5):
        gauss = torch.Tensor(
            [math.exp(-(x - kernel_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(kernel_size)])
        gauss_1D = (gauss / gauss.sum()).unsqueeze(1)
        gauss_2D = gauss_1D.mm(gauss_1D.t()).float().unsqueeze(0).unsqueeze(0)
        return gauss_2D.expand(channel, 1, kernel_size, kernel_size).contiguous()

    def my_cosine_sync(self, x1, x2, x3, kernel_size=3, sigma=1.5):
        nb, ch, row, col = x1.size()
        cosine_1 = F.cosine_similarity(x1, x2, dim=1).unsqueeze(1)
        cosine_2 = F.cosine_similarity(x1, x3, dim=1).unsqueeze(1)
        cosine_3 = F.cosine_similarity(x2, x3, dim=1).unsqueeze(1)
        alpha = self.alpha
        alpha = self.weight_act(alpha)
        weight = alpha / (torch.sum(alpha, dim=0) + self.epsilon)
        cosine = weight[0] * cosine_1 + weight[1] * cosine_2 + weight[2] * cosine_3
        filters = self._gaussian_filter(1, kernel_size, sigma).to(cosine.device)
        attenMap = F.conv2d(cosine, filters, padding=1)
        attenMap = attenMap.expand(nb, ch, row, col).contiguous()
        x1 = x1 * attenMap
        x2 = x2 * attenMap
        x3 = x3 * attenMap
        return x1, x2, x3

    def _status_sync(self, x1, x2, kernel_size=3, sigma=1.5):
        nb, ch, row, col = x1.size()
        cosine = F.cosine_similarity(x1, x2, dim=1).unsqueeze(1)
        filters = self._gaussian_filter(1, kernel_size, sigma).to(cosine.device)
        # print("Cosine =>", cosine.size(), "Filter =>", filters.size())
        attenMap = F.conv2d(cosine, filters, padding=1)
        attenMap = attenMap.expand(nb, ch, row, col).contiguous()
        x1 = x1 * attenMap
        x2 = x2 * attenMap
        return x1, x2

    def forward(self, x):
        if self.src_ch == 1:
            x = torch.cat([x, x, x], dim=1)
        # forward
        x1 = self.stem_conv(x)
        x1 = self.stem_norm(x1)
        x2 = self.relu(x1)

        # Encoder
        x3 = self.layer1(x2)
        x4 = self.layer2(x3)
        x5 = self.layer3(x4)  # Attention
        x6 = self.layer4(x5)  # Attention

        ## stage 1
        x7_R = self.deconvR1(x6)
        x7_G = self.deconvG1(x6)
        x7_B = self.deconvB1(x6)
        # sync
        x7_R, x7_G, x7_B = self.my_cosine_sync(x7_R, x7_G, x7_B)

        x7_R = x7_R + x5
        x7_G = x7_G + x5
        x7_B = x7_B + x5

        ## stage 2
        x8_R = self.upResR1(x7_R)
        x8_R = self.deconvR2(x8_R)
        x8_G = self.upResG1(x7_G)
        x8_G = self.deconvG2(x8_G)
        x8_B = self.upResB1(x7_B)
        x8_B = self.deconvB2(x8_B)
        # sync
        x8_R, x8_G, x8_B = self.my_cosine_sync(x8_R, x8_G, x8_B)

        x8_R = x8_R + x4
        x8_G = x8_G + x4
        x8_B = x8_B + x4

        ## stage 3
        x9_R = self.upResR2(x8_R)
        x9_R = self.deconvR3(x9_R)
        x9_G = self.upResG2(x8_G)
        x9_G = self.deconvG3(x9_G)
        x9_B = self.upResB2(x8_B)
        x9_B = self.deconvB3(x9_B)
        # sync
        x9_R, x9_G, x9_B = self.my_cosine_sync(x9_R, x9_G, x9_B)

        x9_R = x9_R + x3
        x9_G = x9_G + x3
        x9_B = x9_B + x3

        ## stage 4
        x10_R = self.upResR3(x9_R)
        x10_R = self.deconvR4(x10_R)
        x10_G = self.upResG3(x9_G)
        x10_G = self.deconvG4(x10_G)
        x10_B = self.upResB3(x9_B)
        x10_B = self.deconvB4(x10_B)
        # sync
        x10_R, x10_G, x10_B = self.my_cosine_sync(x10_R, x10_G, x10_B)

        ## Final stage
        x_final_R = self.predR(x10_R)
        x_final_G = self.predG(x10_G)
        x_final_B = self.predB(x10_B)

        # x_final = torch.cat((x_final_R, x_final_G, x_final_B), dim=1)

        return x_final_R, x_final_G, x_final_B


class attentionResnetSync_Lab(nn.Module):

    def __init__(self, block, Attention_block, layers, src_ch, tar_ch, BNmode):
        super(attentionResnetSync_Lab, self).__init__()
        kernels = [64, 128, 256, 512]
        self.epsilon = 1e-4
        self.src_ch = src_ch
        self.inplanes = kernels[0]
        self.stem_conv = nn.Conv2d(3, kernels[0], kernel_size=7,
                                   stride=2, padding=3, bias=False)
        print("####1 BNmode => ", BNmode)
        if BNmode == 'BN':
            self.stem_norm = nn.BatchNorm2d(kernels[0])
        elif BNmode == 'IN':
            self.stem_norm = nn.InstanceNorm2d(kernels[0])
        elif BNmode == 'GN':
            self.stem_norm = nn.GroupNorm(32, kernels[0])
        # Adding AdaIN, combined BN and IN
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(block, kernels[0], layers[0], stride=1, BNmode=BNmode)
        self.layer2 = self._make_layer(block, kernels[1], layers[1], stride=2, BNmode=BNmode)
        self.layer3 = self._make_layer(Attention_block, kernels[2], layers[2], stride=2, BNmode=BNmode)
        self.layer4 = self._make_layer(Attention_block, kernels[3], layers[3], stride=2, BNmode=BNmode)

        # Red components
        self.deconvL1 = upconv(kernels[3], kernels[2], ratio="x2")
        self.inplanes = kernels[2]
        self.upResL1 = self._make_layer(block, kernels[2], layers[2], BNmode=BNmode)
        self.deconvL2 = upconv(kernels[2], kernels[1], ratio="x2")
        self.inplanes = kernels[1]
        self.upResL2 = self._make_layer(block, kernels[1], layers[1], BNmode=BNmode)
        self.deconvL3 = upconv(kernels[1], kernels[0], ratio="x2")
        self.inplanes = kernels[0]
        self.upResL3 = self._make_layer(block, kernels[0], layers[0], BNmode=BNmode)
        self.deconvL4 = upconv(kernels[0], kernels[0], ratio="x2")
        self.predL = nn.Conv2d(kernels[0], tar_ch[0], kernel_size=3, stride=1, padding=1, bias=False)

        # a components
        self.deconva1 = upconv(kernels[3], kernels[2], ratio="x2")
        self.inplanes = kernels[2]
        self.upResa1 = self._make_layer(block, kernels[2], layers[2], BNmode=BNmode)
        self.deconva2 = upconv(kernels[2], kernels[1], ratio="x2")
        self.inplanes = kernels[1]
        self.upResa2 = self._make_layer(block, kernels[1], layers[1], BNmode=BNmode)
        self.deconva3 = upconv(kernels[1], kernels[0], ratio="x2")
        self.inplanes = kernels[0]
        self.upResa3 = self._make_layer(block, kernels[0], layers[0], BNmode=BNmode)
        self.deconva4 = upconv(kernels[0], kernels[0], ratio="x2")
        self.preda = nn.Conv2d(kernels[0], tar_ch[1], kernel_size=3, stride=1, padding=1, bias=False)

        # b components
        self.deconvb1 = upconv(kernels[3], kernels[2], ratio="x2")
        self.inplanes = kernels[2]
        self.upResb1 = self._make_layer(block, kernels[2], layers[2], BNmode=BNmode)
        self.deconvb2 = upconv(kernels[2], kernels[1], ratio="x2")
        self.inplanes = kernels[1]
        self.upResb2 = self._make_layer(block, kernels[1], layers[1], BNmode=BNmode)
        self.deconvb3 = upconv(kernels[1], kernels[0], ratio="x2")
        self.inplanes = kernels[0]
        self.upResb3 = self._make_layer(block, kernels[0], layers[0], BNmode=BNmode)
        self.deconvb4 = upconv(kernels[0], kernels[0], ratio="x2")
        self.predb = nn.Conv2d(kernels[0], tar_ch[2], kernel_size=3, stride=1, padding=1, bias=False)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1, BNmode='BN'):
        downsample = None
        print("####2 BNmode => ", BNmode)
        if stride != 1 or self.inplanes != planes * block.expansion:
            bnl = nn.BatchNorm2d(planes * block.expansion)
            if BNmode == 'IN':
                bnl = nn.InstanceNorm2d(planes * block.expansion)
            if BNmode == 'GN':
                bnl = nn.GroupNorm(32, planes * block.expansion)
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                bnl,
            )
            # Adding AdaIN

        layers = []
        if BNmode == 'GN':
            layers.append(block(self.inplanes, planes, stride, downsample, BNmode=BNmode))
        else:
            layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            if BNmode == 'GN':
                layers.append(block(self.inplanes, planes, BNmode=BNmode))
            else:
                layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    @staticmethod
    def _gaussian_filter(channel, kernel_size, sigma=1.5):
        gauss = torch.Tensor(
            [math.exp(-(x - kernel_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(kernel_size)])
        gauss_1D = (gauss / gauss.sum()).unsqueeze(1)
        gauss_2D = gauss_1D.mm(gauss_1D.t()).float().unsqueeze(0).unsqueeze(0)
        return gauss_2D.expand(channel, 1, kernel_size, kernel_size).contiguous()

    def my_cosine_sync(self, x1, x2, kernel_size=3, sigma=1.5):
        nb, ch, row, col = x1.size()
        cosine = F.cosine_similarity(x1, x2, dim=1).unsqueeze(1)
        filters = self._gaussian_filter(1, kernel_size, sigma).to(cosine.device)
        attenMap = F.conv2d(cosine, filters, padding=1)
        attenMap = attenMap.expand(nb, ch, row, col).contiguous()
        x1 = x1 * attenMap
        x2 = x2 * attenMap
        return x1, x2

    def _status_sync(self, x1, x2, kernel_size=3, sigma=1.5):
        nb, ch, row, col = x1.size()
        cosine = F.cosine_similarity(x1, x2, dim=1).unsqueeze(1)
        filters = self._gaussian_filter(1, kernel_size, sigma).to(cosine.device)
        # print("Cosine =>", cosine.size(), "Filter =>", filters.size())
        attenMap = F.conv2d(cosine, filters, padding=1)
        attenMap = attenMap.expand(nb, ch, row, col).contiguous()
        x1 = x1 * attenMap
        x2 = x2 * attenMap
        return x1, x2

    def forward(self, x):
        if self.src_ch == 1:
            x = torch.cat([x, x, x], dim=1)
        # forward
        x1 = self.stem_conv(x)
        x1 = self.stem_norm(x1)
        x2 = self.relu(x1)

        # Encoder
        x3 = self.layer1(x2)
        x4 = self.layer2(x3)
        x5 = self.layer3(x4)  # Attention
        x6 = self.layer4(x5)  # Attention

        ## stage 1
        x7_L = self.deconvL1(x6)
        x7_a = self.deconva1(x6)
        x7_b = self.deconvb1(x6)
        # sync
        x7_a, x7_b = self.my_cosine_sync(x7_a, x7_b)

        x7_L = x7_L + x5
        x7_a = x7_a + x5
        x7_b = x7_b + x5

        ## stage 2
        x8_L = self.upResL1(x7_L)
        x8_L = self.deconvL2(x8_L)
        x8_a = self.upResa1(x7_a)
        x8_a = self.deconva2(x8_a)
        x8_b = self.upResb1(x7_b)
        x8_b = self.deconvb2(x8_b)
        # sync
        x8_a, x8_b = self.my_cosine_sync(x8_a, x8_b)

        x8_L = x8_L + x4
        x8_a = x8_a + x4
        x8_b = x8_b + x4

        ## stage 3
        x9_L = self.upResL2(x8_L)
        x9_L = self.deconvL3(x9_L)
        x9_a = self.upResa2(x8_a)
        x9_a = self.deconva3(x9_a)
        x9_b = self.upResb2(x8_b)
        x9_b = self.deconvb3(x9_b)
        # sync
        x9_a, x9_b = self.my_cosine_sync(x9_a, x9_b)

        x9_L = x9_L + x3
        x9_a = x9_a + x3
        x9_b = x9_b + x3

        ## stage 4
        x10_L = self.upResL3(x9_L)
        x10_L = self.deconvL4(x10_L)
        x10_a = self.upResa3(x9_a)
        x10_a = self.deconva4(x10_a)
        x10_b = self.upResb3(x9_b)
        x10_b = self.deconvb4(x10_b)
        # sync
        x10_a, x10_b = self.my_cosine_sync(x10_a, x10_b)

        ## Final stage
        x_final_L = self.predL(x10_L)
        x_final_a = self.preda(x10_a)
        x_final_b = self.predb(x10_b)

        return x_final_L, x_final_a, x_final_b


class Resnet_VAE(nn.Module):

    def __init__(self, block, layers, src_ch, tar_ch, BNmode, z_dim=100):
        super(Resnet_VAE, self).__init__()
        kernels = [64, 128, 256, 512]
        self.z_dim = z_dim
        self.alpha = nn.Parameter(torch.ones(2))
        self.weight_act = nn.ReLU()
        self.src_ch = src_ch
        self.inplanes = kernels[0]
        self.stem_conv = nn.Conv2d(3, kernels[0], kernel_size=7,
                                   stride=2, padding=3, bias=False)
        print("####1 BNmode => ", BNmode)
        if BNmode == 'BN':
            self.stem_norm = nn.BatchNorm2d(kernels[0])
        elif BNmode == 'IN':
            self.stem_norm = nn.InstanceNorm2d(kernels[0])
        elif BNmode == 'GN':
            self.stem_norm = nn.GroupNorm(32, kernels[0])
        # Adding AdaIN, combined BN and IN
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(block, kernels[0], layers[0], stride=1, BNmode=BNmode)
        self.layer2 = self._make_layer(block, kernels[1], layers[1], stride=2, BNmode=BNmode)
        self.layer3 = self._make_layer(block, kernels[2], layers[2], stride=2, BNmode=BNmode)
        self.layer4 = self._make_layer(block, kernels[3], layers[3], stride=2, BNmode=BNmode)
        self.linear1 = nn.Linear(kernels[3], 2 * z_dim)

        # Red components
        self.linear2 = nn.Linear(z_dim, kernels[3])
        self.deconvL1 = upconv(kernels[3], kernels[2], ratio="x2")
        self.inplanes = kernels[2]
        self.upResL1 = self._make_layer(block, kernels[2], layers[2], BNmode=BNmode)
        self.deconvL2 = upconv(kernels[2], kernels[1], ratio="x2")
        self.inplanes = kernels[1]
        self.upResL2 = self._make_layer(block, kernels[1], layers[1], BNmode=BNmode)
        self.deconvL3 = upconv(kernels[1], kernels[0], ratio="x2")
        self.inplanes = kernels[0]
        self.upResL3 = self._make_layer(block, kernels[0], layers[0], BNmode=BNmode)
        self.deconvL4 = upconv(kernels[0], kernels[0], ratio="x2")
        self.predL = nn.Conv2d(kernels[0], tar_ch[0], kernel_size=3, stride=1, padding=1, bias=False)

        # a components
        self.deconva1 = upconv(kernels[3], kernels[2], ratio="x2")
        self.inplanes = kernels[2]
        self.upResa1 = self._make_layer(block, kernels[2], layers[2], BNmode=BNmode)
        self.deconva2 = upconv(kernels[2], kernels[1], ratio="x2")
        self.inplanes = kernels[1]
        self.upResa2 = self._make_layer(block, kernels[1], layers[1], BNmode=BNmode)
        self.deconva3 = upconv(kernels[1], kernels[0], ratio="x2")
        self.inplanes = kernels[0]
        self.upResa3 = self._make_layer(block, kernels[0], layers[0], BNmode=BNmode)
        self.deconva4 = upconv(kernels[0], kernels[0], ratio="x2")
        self.preda = nn.Conv2d(kernels[0], tar_ch[1], kernel_size=3, stride=1, padding=1, bias=False)

        # b components
        self.deconvb1 = upconv(kernels[3], kernels[2], ratio="x2")
        self.inplanes = kernels[2]
        self.upResb1 = self._make_layer(block, kernels[2], layers[2], BNmode=BNmode)
        self.deconvb2 = upconv(kernels[2], kernels[1], ratio="x2")
        self.inplanes = kernels[1]
        self.upResb2 = self._make_layer(block, kernels[1], layers[1], BNmode=BNmode)
        self.deconvb3 = upconv(kernels[1], kernels[0], ratio="x2")
        self.inplanes = kernels[0]
        self.upResb3 = self._make_layer(block, kernels[0], layers[0], BNmode=BNmode)
        self.deconvb4 = upconv(kernels[0], kernels[0], ratio="x2")
        self.predb = nn.Conv2d(kernels[0], tar_ch[2], kernel_size=3, stride=1, padding=1, bias=False)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1, BNmode='BN'):
        downsample = None
        print("####2 BNmode => ", BNmode)
        if stride != 1 or self.inplanes != planes * block.expansion:
            bnl = nn.BatchNorm2d(planes * block.expansion)
            if BNmode == 'IN':
                bnl = nn.InstanceNorm2d(planes * block.expansion)
            if BNmode == 'GN':
                bnl = nn.GroupNorm(32, planes * block.expansion)
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                bnl,
            )
            # Adding AdaIN

        layers = []
        if BNmode == 'GN':
            layers.append(block(self.inplanes, planes, stride, downsample, BNmode=BNmode))
        else:
            layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            if BNmode == 'GN':
                layers.append(block(self.inplanes, planes, BNmode=BNmode))
            else:
                layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    @staticmethod
    def _gaussian_filter(channel, kernel_size, sigma=1.5):
        gauss = torch.Tensor(
            [math.exp(-(x - kernel_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(kernel_size)])
        gauss_1D = (gauss / gauss.sum()).unsqueeze(1)
        gauss_2D = gauss_1D.mm(gauss_1D.t()).float().unsqueeze(0).unsqueeze(0)
        return gauss_2D.expand(channel, 1, kernel_size, kernel_size).contiguous()

    def my_cosine_sync(self, x1, x2, x3, kernel_size=3, sigma=1.5):
        nb, ch, row, col = x1.size()
        cosine_1 = F.cosine_similarity(x1, x2, dim=1).unsqueeze(1)
        cosine_2 = F.cosine_similarity(x1, x3, dim=1).unsqueeze(1)
        cosine = cosine_1 + cosine_2
        filters = self._gaussian_filter(1, kernel_size, sigma).to(cosine.device)
        attenMap = F.conv2d(cosine, filters, padding=1)
        attenMap = attenMap.expand(nb, ch, row, col).contiguous()
        x2 = x2 * attenMap
        x3 = x3 * attenMap
        return x1, x2, x3

    def _status_sync(self, x1, x2, kernel_size=3, sigma=1.5):
        nb, ch, row, col = x1.size()
        cosine = F.cosine_similarity(x1, x2, dim=1).unsqueeze(1)
        filters = self._gaussian_filter(1, kernel_size, sigma).to(cosine.device)
        # print("Cosine =>", cosine.size(), "Filter =>", filters.size())
        attenMap = F.conv2d(cosine, filters, padding=1)
        attenMap = attenMap.expand(nb, ch, row, col).contiguous()
        x1 = x1 * attenMap
        x2 = x2 * attenMap
        return x1, x2

    def reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        eps = Variable(std.data.new(std.size()).normal_())
        # return eps.mul(std).add_(mu)
        return mu

    def forward(self, x):
        if self.src_ch == 1:
            x = torch.cat([x, x, x], dim=1)
        # forward
        x1 = self.stem_conv(x)
        x1 = self.stem_norm(x1)
        x2 = self.relu(x1)

        # Encoder
        x3 = self.layer1(x2)
        x4 = self.layer2(x3)
        x5 = self.layer3(x4)
        x6 = self.layer4(x5)
        x6 = F.adaptive_avg_pool2d(x6, 1)
        x6 = x6.view(x6.size(0), -1)
        x6 = self.linear1(x6)
        mu = x6[:, :self.z_dim]
        logvar = x6[:, self.z_dim:]

        # Bottleneck
        z = self.reparameterize(mu, logvar)

        ## stage 1
        z_bottle = self.linear2(z)
        z_bottle = z_bottle.view(z.size(0), 512, 1, 1)
        z_bottle = F.interpolate(z_bottle, size=(x5.size(2) // 2, x5.size(3) // 2))

        x7_L = self.deconvL1(z_bottle)
        x7_a = self.deconva1(z_bottle)
        x7_b = self.deconvb1(z_bottle)

        # sync
        x7_L, x7_a, x7_b = self.my_cosine_sync(x7_L, x7_a, x7_b)

        x7_L = x7_L + x5
        x7_a = x7_a + x5
        x7_b = x7_b + x5

        ## stage 2
        x8_L = self.upResL1(x7_L)
        x8_L = self.deconvL2(x8_L)
        x8_a = self.upResa1(x7_a)
        x8_a = self.deconva2(x8_a)
        x8_b = self.upResb1(x7_b)
        x8_b = self.deconvb2(x8_b)
        # sync
        x8_L, x8_a, x8_b = self.my_cosine_sync(x8_L, x8_a, x8_b)

        x8_L = x8_L + x4
        x8_a = x8_a + x4
        x8_b = x8_b + x4

        ## stage 3
        x9_L = self.upResL2(x8_L)
        x9_L = self.deconvL3(x9_L)
        x9_a = self.upResa2(x8_a)
        x9_a = self.deconva3(x9_a)
        x9_b = self.upResb2(x8_b)
        x9_b = self.deconvb3(x9_b)
        # sync
        x9_L, x9_a, x9_b = self.my_cosine_sync(x9_L, x9_a, x9_b)

        x9_L = x9_L + x3
        x9_a = x9_a + x3
        x9_b = x9_b + x3

        ## stage 4
        x10_L = self.upResL3(x9_L)
        x10_L = self.deconvL4(x10_L)
        x10_a = self.upResa3(x9_a)
        x10_a = self.deconva4(x10_a)
        x10_b = self.upResb3(x9_b)
        x10_b = self.deconvb4(x10_b)
        # sync
        x10_L, x10_a, x10_b = self.my_cosine_sync(x10_L, x10_a, x10_b)

        ## Final stage
        x_final_L = self.predL(x10_L)
        x_final_a = self.preda(x10_a)
        x_final_b = self.predb(x10_b)

        return x_final_L, x_final_a, x_final_b, mu, logvar


class GaussianBlurConv(nn.Module):
    def __init__(self, channels):
        super(GaussianBlurConv, self).__init__()
        self.channels = channels
        kernel = [[0.00078633, 0.00655965, 0.01330373, 0.00655965, 0.00078633],
                  [0.00655965, 0.05472157, 0.11098164, 0.05472157, 0.00655965],
                  [0.01330373, 0.11098164, 0.22508352, 0.11098164, 0.01330373],
                  [0.00655965, 0.05472157, 0.11098164, 0.05472157, 0.00655965],
                  [0.00078633, 0.00655965, 0.01330373, 0.00655965, 0.00078633]]
        kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0)
        kernel = np.repeat(kernel, self.channels, axis=0)
        self.weight = nn.Parameter(data=kernel, requires_grad=False)

    def __call__(self, x):
        x = F.conv2d(x, self.weight, padding=2, groups=self.channels)
        return x


class ResNet_AAE(nn.Module):

    def __init__(self, block, layers, src_ch, tar_ch, BNmode, z_dim=100):
        super(ResNet_AAE, self).__init__()
        kernels = [64, 128, 256, 512]
        self.z_dim = z_dim
        self.epsilon = 1e-4
        self.src_ch = src_ch
        self.inplanes = kernels[0]
        self.stem_conv = nn.Conv2d(3, kernels[0], kernel_size=7,
                                   stride=2, padding=3, bias=False)
        print("####1 BNmode => ", BNmode)
        if BNmode == 'BN':
            self.stem_norm = nn.BatchNorm2d(kernels[0])
        elif BNmode == 'IN':
            self.stem_norm = nn.InstanceNorm2d(kernels[0])
        elif BNmode == 'GN':
            self.stem_norm = nn.GroupNorm(32, kernels[0])
        # Adding AdaIN, combined BN and IN
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(block, kernels[0], layers[0], stride=1, BNmode=BNmode)
        self.layer2 = self._make_layer(block, kernels[1], layers[1], stride=2, BNmode=BNmode)
        self.layer3 = self._make_layer(block, kernels[2], layers[2], stride=2, BNmode=BNmode)
        self.layer4 = self._make_layer(block, kernels[3], layers[3], stride=2, BNmode=BNmode)
        self.linear1 = nn.Linear(kernels[3], 2 * z_dim)  # encoding latent code z

        self.linear2 = nn.Linear(z_dim, kernels[3])  # decodng latent code z
        # Red components
        self.deconvL1 = upconv(kernels[3], kernels[2], ratio="x2")
        self.inplanes = kernels[2]
        self.upResL1 = self._make_layer(block, kernels[2], layers[2], BNmode=BNmode)
        self.deconvL2 = upconv(kernels[2], kernels[1], ratio="x2")
        self.inplanes = kernels[1]
        self.upResL2 = self._make_layer(block, kernels[1], layers[1], BNmode=BNmode)
        self.deconvL3 = upconv(kernels[1], kernels[0], ratio="x2")
        self.inplanes = kernels[0]
        self.upResL3 = self._make_layer(block, kernels[0], layers[0], BNmode=BNmode)
        self.deconvL4 = upconv(kernels[0], kernels[0], ratio="x2")
        self.predL = nn.Conv2d(kernels[0], tar_ch[0], kernel_size=3, stride=1, padding=1, bias=False)

        # a components
        self.deconva1 = upconv(kernels[3], kernels[2], ratio="x2")
        self.inplanes = kernels[2]
        self.upResa1 = self._make_layer(block, kernels[2], layers[2], BNmode=BNmode)
        self.deconva2 = upconv(kernels[2], kernels[1], ratio="x2")
        self.inplanes = kernels[1]
        self.upResa2 = self._make_layer(block, kernels[1], layers[1], BNmode=BNmode)
        self.deconva3 = upconv(kernels[1], kernels[0], ratio="x2")
        self.inplanes = kernels[0]
        self.upResa3 = self._make_layer(block, kernels[0], layers[0], BNmode=BNmode)
        self.deconva4 = upconv(kernels[0], kernels[0], ratio="x2")
        self.preda = nn.Conv2d(kernels[0], tar_ch[1], kernel_size=3, stride=1, padding=1, bias=False)

        # b components
        self.deconvb1 = upconv(kernels[3], kernels[2], ratio="x2")
        self.inplanes = kernels[2]
        self.upResb1 = self._make_layer(block, kernels[2], layers[2], BNmode=BNmode)
        self.deconvb2 = upconv(kernels[2], kernels[1], ratio="x2")
        self.inplanes = kernels[1]
        self.upResb2 = self._make_layer(block, kernels[1], layers[1], BNmode=BNmode)
        self.deconvb3 = upconv(kernels[1], kernels[0], ratio="x2")
        self.inplanes = kernels[0]
        self.upResb3 = self._make_layer(block, kernels[0], layers[0], BNmode=BNmode)
        self.deconvb4 = upconv(kernels[0], kernels[0], ratio="x2")
        self.predb = nn.Conv2d(kernels[0], tar_ch[2], kernel_size=3, stride=1, padding=1, bias=False)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1, BNmode='BN'):
        downsample = None
        print("####2 BNmode => ", BNmode)
        if stride != 1 or self.inplanes != planes * block.expansion:
            bnl = nn.BatchNorm2d(planes * block.expansion)
            if BNmode == 'IN':
                bnl = nn.InstanceNorm2d(planes * block.expansion)
            if BNmode == 'GN':
                bnl = nn.GroupNorm(32, planes * block.expansion)
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                bnl,
            )
            # Adding AdaIN

        layers = []
        if BNmode == 'GN':
            layers.append(block(self.inplanes, planes, stride, downsample, BNmode=BNmode))
        else:
            layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            if BNmode == 'GN':
                layers.append(block(self.inplanes, planes, BNmode=BNmode))
            else:
                layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    @staticmethod
    def _gaussian_filter(channel, kernel_size, sigma=1.5):
        gauss = torch.Tensor(
            [math.exp(-(x - kernel_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(kernel_size)])
        gauss_1D = (gauss / gauss.sum()).unsqueeze(1)
        gauss_2D = gauss_1D.mm(gauss_1D.t()).float().unsqueeze(0).unsqueeze(0)
        return gauss_2D.expand(channel, 1, kernel_size, kernel_size).contiguous()

    @staticmethod
    def my_gaussian_filter(channels, kernel_size=15, sigma=3):
        kernel_size = kernel_size  # 15
        sigma = sigma

        x_cord = torch.arange(kernel_size)
        x_grid = x_cord.repeat(kernel_size).view(kernel_size, kernel_size)
        y_grid = x_grid.t()
        xy_grid = torch.stack([x_grid, y_grid], dim=-1)

        mean = (kernel_size - 1) / 2.
        variance = sigma ** 2

        gaussian_kernel = (1. / (2. * math.pi * variance)) * \
                          torch.exp(-torch.sum((xy_grid - mean) ** 2., dim=-1) / \
                                    (2 * variance))
        gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)
        gaussian_kernel = gaussian_kernel.repeat(channels, 1, 1, 1)

        return gaussian_kernel

    def my_cosine_sync(self, x1, x2, x3, kernel_size=3, sigma=1.5):
        nb, ch, row, col = x1.size()
        # gaussian_filter = self._gaussian_filter(1, kernel_size, sigma).to(x1.device)
        cosine_1 = F.cosine_similarity(x1, x2, dim=1).unsqueeze(1)
        cosine_2 = F.cosine_similarity(x1, x3, dim=1).unsqueeze(1)
        # gaussian_filter = nn.Conv2d(in_channels=ch, out_channels=ch,
        #                           kernel_size=15, groups=ch, bias=False, padding=2)
        # gaussian_filter.weight.data = self.my_gaussian_filter(channels=ch)
        # gaussian_filter.weight.requires_grad = False

        # gaussian_edge = x1 - gaussian_filter.cuda()(x1)
        gaussian_conv = GaussianBlurConv(channels=ch).cuda()
        gaussian_blur = gaussian_conv(x1)
        gaussian_edge = x1 - gaussian_blur
        conv_1x1 = nn.Conv2d(in_channels=ch, out_channels=1, kernel_size=1).cuda()
        gaussian_edge = conv_1x1(gaussian_edge)
        concat = torch.cat([gaussian_edge, cosine_1, cosine_2], dim=1)
        conv1d = nn.Conv2d(in_channels=3, out_channels=1, kernel_size=1).cuda()
        attenMap = conv1d(concat)
        attenMap = attenMap.expand(nb, ch, row, col).contiguous()
        x1 = x1 * attenMap
        x2 = x2 * attenMap
        x3 = x3 * attenMap
        return x1, x2, x3

    def _status_sync(self, x1, x2, kernel_size=3, sigma=1.5):
        nb, ch, row, col = x1.size()
        cosine = F.cosine_similarity(x1, x2, dim=1).unsqueeze(1)
        filters = self._gaussian_filter(1, kernel_size, sigma).to(cosine.device)
        # print("Cosine =>", cosine.size(), "Filter =>", filters.size())
        attenMap = F.conv2d(cosine, filters, padding=1)
        attenMap = attenMap.expand(nb, ch, row, col).contiguous()
        x1 = x1 * attenMap
        x2 = x2 * attenMap
        return x1, x2

    def reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        eps = Variable(std.data.new(std.size()).normal_())
        return eps.mul(std).add_(mu)
        # return mu

    def forward(self, x):
        if self.src_ch == 1:
            x = torch.cat([x, x, x], dim=1)
        # forward
        x1 = self.stem_conv(x)
        x1 = self.stem_norm(x1)
        x2 = self.relu(x1)

        # Encoder
        x3 = self.layer1(x2)
        x4 = self.layer2(x3)
        x5 = self.layer3(x4)
        x6 = self.layer4(x5)
        x6 = F.adaptive_avg_pool2d(x6, 1)
        x6 = x6.view(x6.size(0), -1)
        x6 = self.linear1(x6)
        mu = x6[:, :self.z_dim]
        logvar = x6[:, self.z_dim:]

        # Bottleneck
        z = self.reparameterize(mu, logvar)

        ## stage 1
        z_bottle = self.linear2(z)
        z_bottle = z_bottle.view(z.size(0), 512, 1, 1)
        z_bottle = F.interpolate(z_bottle, size=(x5.size(2) // 2, x5.size(3) // 2))

        x7_L = self.deconvL1(z_bottle)
        x7_a = self.deconva1(z_bottle)
        x7_b = self.deconvb1(z_bottle)

        # sync
        x7_L, x7_a, x7_b = self.my_cosine_sync(x7_L, x7_a, x7_b)

        x7_L = x7_L + x5
        x7_a = x7_a + x5
        x7_b = x7_b + x5

        ## stage 2
        x8_L = self.upResL1(x7_L)
        x8_L = self.deconvL2(x8_L)
        x8_a = self.upResa1(x7_a)
        x8_a = self.deconva2(x8_a)
        x8_b = self.upResb1(x7_b)
        x8_b = self.deconvb2(x8_b)
        # sync
        x8_L, x8_a, x8_b = self.my_cosine_sync(x8_L, x8_a, x8_b)

        x8_L = x8_L + x4
        x8_a = x8_a + x4
        x8_b = x8_b + x4

        ## stage 3
        x9_L = self.upResL2(x8_L)
        x9_L = self.deconvL3(x9_L)
        x9_a = self.upResa2(x8_a)
        x9_a = self.deconva3(x9_a)
        x9_b = self.upResb2(x8_b)
        x9_b = self.deconvb3(x9_b)
        # sync
        x9_L, x9_a, x9_b = self.my_cosine_sync(x9_L, x9_a, x9_b)

        x9_L = x9_L + x3
        x9_a = x9_a + x3
        x9_b = x9_b + x3

        ## stage 4
        x10_L = self.upResL3(x9_L)
        x10_L = self.deconvL4(x10_L)
        x10_a = self.upResa3(x9_a)
        x10_a = self.deconva4(x10_a)
        x10_b = self.upResb3(x9_b)
        x10_b = self.deconvb4(x10_b)
        # sync
        x10_L, x10_a, x10_b = self.my_cosine_sync(x10_L, x10_a, x10_b)

        ## Final stage
        x_final_L = self.predL(x10_L)
        x_final_a = self.preda(x10_a)
        x_final_b = self.predb(x10_b)

        return x_final_L, x_final_a, x_final_b, mu, logvar


class ResNet_AAE(nn.Module):

    def __init__(self, block, layers, src_ch, tar_ch, BNmode, z_dim=100):
        super(ResNet_AAE, self).__init__()
        kernels = [64, 128, 256, 512]
        self.z_dim = z_dim
        self.epsilon = 1e-4
        self.src_ch = src_ch
        self.inplanes = kernels[0]
        self.stem_conv = nn.Conv2d(3, kernels[0], kernel_size=7,
                                   stride=2, padding=3, bias=False)
        print("####1 BNmode => ", BNmode)
        if BNmode == 'BN':
            self.stem_norm = nn.BatchNorm2d(kernels[0])
        elif BNmode == 'IN':
            self.stem_norm = nn.InstanceNorm2d(kernels[0])
        elif BNmode == 'GN':
            self.stem_norm = nn.GroupNorm(32, kernels[0])
        # Adding AdaIN, combined BN and IN
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(block, kernels[0], layers[0], stride=1, BNmode=BNmode)
        self.layer2 = self._make_layer(block, kernels[1], layers[1], stride=2, BNmode=BNmode)
        self.layer3 = self._make_layer(block, kernels[2], layers[2], stride=2, BNmode=BNmode)
        self.layer4 = self._make_layer(block, kernels[3], layers[3], stride=2, BNmode=BNmode)
        self.linear1 = nn.Linear(kernels[3], 2 * z_dim)  # encoding latent code z

        self.linear2 = nn.Linear(z_dim, kernels[3])  # decodng latent code z
        # Red components
        self.deconvL1 = upconv(kernels[3], kernels[2], ratio="x2")
        self.inplanes = kernels[2]
        self.upResL1 = self._make_layer(block, kernels[2], layers[2], BNmode=BNmode)
        self.deconvL2 = upconv(kernels[2], kernels[1], ratio="x2")
        self.inplanes = kernels[1]
        self.upResL2 = self._make_layer(block, kernels[1], layers[1], BNmode=BNmode)
        self.deconvL3 = upconv(kernels[1], kernels[0], ratio="x2")
        self.inplanes = kernels[0]
        self.upResL3 = self._make_layer(block, kernels[0], layers[0], BNmode=BNmode)
        self.deconvL4 = upconv(kernels[0], kernels[0], ratio="x2")
        self.predL = nn.Conv2d(kernels[0], tar_ch[0], kernel_size=3, stride=1, padding=1, bias=False)

        # a components
        self.deconva1 = upconv(kernels[3], kernels[2], ratio="x2")
        self.inplanes = kernels[2]
        self.upResa1 = self._make_layer(block, kernels[2], layers[2], BNmode=BNmode)
        self.deconva2 = upconv(kernels[2], kernels[1], ratio="x2")
        self.inplanes = kernels[1]
        self.upResa2 = self._make_layer(block, kernels[1], layers[1], BNmode=BNmode)
        self.deconva3 = upconv(kernels[1], kernels[0], ratio="x2")
        self.inplanes = kernels[0]
        self.upResa3 = self._make_layer(block, kernels[0], layers[0], BNmode=BNmode)
        self.deconva4 = upconv(kernels[0], kernels[0], ratio="x2")
        self.preda = nn.Conv2d(kernels[0], tar_ch[1], kernel_size=3, stride=1, padding=1, bias=False)

        # b components
        self.deconvb1 = upconv(kernels[3], kernels[2], ratio="x2")
        self.inplanes = kernels[2]
        self.upResb1 = self._make_layer(block, kernels[2], layers[2], BNmode=BNmode)
        self.deconvb2 = upconv(kernels[2], kernels[1], ratio="x2")
        self.inplanes = kernels[1]
        self.upResb2 = self._make_layer(block, kernels[1], layers[1], BNmode=BNmode)
        self.deconvb3 = upconv(kernels[1], kernels[0], ratio="x2")
        self.inplanes = kernels[0]
        self.upResb3 = self._make_layer(block, kernels[0], layers[0], BNmode=BNmode)
        self.deconvb4 = upconv(kernels[0], kernels[0], ratio="x2")
        self.predb = nn.Conv2d(kernels[0], tar_ch[2], kernel_size=3, stride=1, padding=1, bias=False)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1, BNmode='BN'):
        downsample = None
        print("####2 BNmode => ", BNmode)
        if stride != 1 or self.inplanes != planes * block.expansion:
            bnl = nn.BatchNorm2d(planes * block.expansion)
            if BNmode == 'IN':
                bnl = nn.InstanceNorm2d(planes * block.expansion)
            if BNmode == 'GN':
                bnl = nn.GroupNorm(32, planes * block.expansion)
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                bnl,
            )
            # Adding AdaIN

        layers = []
        if BNmode == 'GN':
            layers.append(block(self.inplanes, planes, stride, downsample, BNmode=BNmode))
        else:
            layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            if BNmode == 'GN':
                layers.append(block(self.inplanes, planes, BNmode=BNmode))
            else:
                layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    @staticmethod
    def _gaussian_filter(channel, kernel_size, sigma=1.5):
        gauss = torch.Tensor(
            [math.exp(-(x - kernel_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(kernel_size)])
        gauss_1D = (gauss / gauss.sum()).unsqueeze(1)
        gauss_2D = gauss_1D.mm(gauss_1D.t()).float().unsqueeze(0).unsqueeze(0)
        return gauss_2D.expand(channel, 1, kernel_size, kernel_size).contiguous()

    @staticmethod
    def my_gaussian_filter(channels, kernel_size=15, sigma=3):
        kernel_size = kernel_size  # 15
        sigma = sigma

        x_cord = torch.arange(kernel_size)
        x_grid = x_cord.repeat(kernel_size).view(kernel_size, kernel_size)
        y_grid = x_grid.t()
        xy_grid = torch.stack([x_grid, y_grid], dim=-1)

        mean = (kernel_size - 1) / 2.
        variance = sigma ** 2

        gaussian_kernel = (1. / (2. * math.pi * variance)) * \
                          torch.exp(-torch.sum((xy_grid - mean) ** 2., dim=-1) / \
                                    (2 * variance))
        gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)
        gaussian_kernel = gaussian_kernel.repeat(channels, 1, 1, 1)

        return gaussian_kernel

    def my_cosine_sync(self, x1, x2, x3, kernel_size=3, sigma=1.5):
        nb, ch, row, col = x1.size()
        # gaussian_filter = self._gaussian_filter(1, kernel_size, sigma).to(x1.device)
        cosine_1 = F.cosine_similarity(x1, x2, dim=1).unsqueeze(1)
        cosine_2 = F.cosine_similarity(x1, x3, dim=1).unsqueeze(1)
        # gaussian_filter = nn.Conv2d(in_channels=ch, out_channels=ch,
        #                           kernel_size=15, groups=ch, bias=False, padding=2)
        # gaussian_filter.weight.data = self.my_gaussian_filter(channels=ch)
        # gaussian_filter.weight.requires_grad = False

        # gaussian_edge = x1 - gaussian_filter.cuda()(x1)
        gaussian_conv = GaussianBlurConv(channels=ch).cuda()
        gaussian_blur = gaussian_conv(x1)
        gaussian_edge = x1 - gaussian_blur
        conv_1x1 = nn.Conv2d(in_channels=ch, out_channels=1, kernel_size=1).cuda()
        gaussian_edge = conv_1x1(gaussian_edge)
        concat = torch.cat([gaussian_edge, cosine_1, cosine_2], dim=1)
        conv1d = nn.Conv2d(in_channels=3, out_channels=1, kernel_size=1).cuda()
        attenMap = conv1d(concat)
        attenMap = attenMap.expand(nb, ch, row, col).contiguous()
        x1 = x1 * attenMap
        x2 = x2 * attenMap
        x3 = x3 * attenMap
        return x1, x2, x3

    def _status_sync(self, x1, x2, kernel_size=3, sigma=1.5):
        nb, ch, row, col = x1.size()
        cosine = F.cosine_similarity(x1, x2, dim=1).unsqueeze(1)
        filters = self._gaussian_filter(1, kernel_size, sigma).to(cosine.device)
        # print("Cosine =>", cosine.size(), "Filter =>", filters.size())
        attenMap = F.conv2d(cosine, filters, padding=1)
        attenMap = attenMap.expand(nb, ch, row, col).contiguous()
        x1 = x1 * attenMap
        x2 = x2 * attenMap
        return x1, x2

    def reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        eps = Variable(std.data.new(std.size()).normal_())
        return eps.mul(std).add_(mu)
        # return mu

    def forward(self, x):
        if self.src_ch == 1:
            x = torch.cat([x, x, x], dim=1)
        # forward
        x1 = self.stem_conv(x)
        x1 = self.stem_norm(x1)
        x2 = self.relu(x1)

        # Encoder
        x3 = self.layer1(x2)
        x4 = self.layer2(x3)
        x5 = self.layer3(x4)
        x6 = self.layer4(x5)
        x6 = F.adaptive_avg_pool2d(x6, 1)
        x6 = x6.view(x6.size(0), -1)
        x6 = self.linear1(x6)
        mu = x6[:, :self.z_dim]
        logvar = x6[:, self.z_dim:]

        # Bottleneck
        z = self.reparameterize(mu, logvar)

        ## stage 1
        z_bottle = self.linear2(z)
        z_bottle = z_bottle.view(z.size(0), 512, 1, 1)
        z_bottle = F.interpolate(z_bottle, size=(x5.size(2) // 2, x5.size(3) // 2))

        x7_L = self.deconvL1(z_bottle)
        x7_a = self.deconva1(z_bottle)
        x7_b = self.deconvb1(z_bottle)

        # sync
        x7_L, x7_a, x7_b = self.my_cosine_sync(x7_L, x7_a, x7_b)

        x7_L = x7_L + x5
        x7_a = x7_a + x5
        x7_b = x7_b + x5

        ## stage 2
        x8_L = self.upResL1(x7_L)
        x8_L = self.deconvL2(x8_L)
        x8_a = self.upResa1(x7_a)
        x8_a = self.deconva2(x8_a)
        x8_b = self.upResb1(x7_b)
        x8_b = self.deconvb2(x8_b)
        # sync
        x8_L, x8_a, x8_b = self.my_cosine_sync(x8_L, x8_a, x8_b)

        x8_L = x8_L + x4
        x8_a = x8_a + x4
        x8_b = x8_b + x4

        ## stage 3
        x9_L = self.upResL2(x8_L)
        x9_L = self.deconvL3(x9_L)
        x9_a = self.upResa2(x8_a)
        x9_a = self.deconva3(x9_a)
        x9_b = self.upResb2(x8_b)
        x9_b = self.deconvb3(x9_b)
        # sync
        x9_L, x9_a, x9_b = self.my_cosine_sync(x9_L, x9_a, x9_b)

        x9_L = x9_L + x3
        x9_a = x9_a + x3
        x9_b = x9_b + x3

        ## stage 4
        x10_L = self.upResL3(x9_L)
        x10_L = self.deconvL4(x10_L)
        x10_a = self.upResa3(x9_a)
        x10_a = self.deconva4(x10_a)
        x10_b = self.upResb3(x9_b)
        x10_b = self.deconvb4(x10_b)
        # sync
        x10_L, x10_a, x10_b = self.my_cosine_sync(x10_L, x10_a, x10_b)

        ## Final stage
        x_final_L = self.predL(x10_L)
        x_final_a = self.preda(x10_a)
        x_final_b = self.predb(x10_b)

        return x_final_L, x_final_a, x_final_b, mu, logvar


class Quantize(nn.Module):
    def __init__(self, dim, n_embed, decay=0.99, eps=1e-5):
        super().__init__()
        self.dim = dim
        self.n_embed = n_embed
        self.decay = decay
        self.eps = eps

        embed = torch.randn(dim, n_embed)
        self.register_buffer("embed", embed)
        self.register_buffer("cluster_size", torch.zeros(n_embed))
        self.register_buffer("embed_avg", embed.clone())

    def forward(self, input):
        flatten = input.reshape(-1, self.dim)
        dist = (
                flatten.pow(2).sum(1, keepdim=True)
                - 2 * flatten @ self.embed
                + self.embed.pow(2).sum(0, keepdim=True)
        )
        _, embed_ind = (-dist).max(1)
        embed_onehot = F.one_hot(embed_ind, self.n_embed).type(flatten.dtype)
        embed_ind = embed_ind.view(*input.shape[:-1])
        quantize = self.embed_code(embed_ind)

        if self.training:
            embed_onehot_sum = embed_onehot.sum(0)
            embed_sum = flatten.transpose(0, 1) @ embed_onehot

            dist_fn.all_reduce(embed_onehot_sum)
            dist_fn.all_reduce(embed_sum)

            self.cluster_size.data.mul_(self.decay).add_(
                embed_onehot_sum, alpha=1 - self.decay
            )
            self.embed_avg.data.mul_(self.decay).add_(embed_sum, alpha=1 - self.decay)
            n = self.cluster_size.sum()
            cluster_size = (
                    (self.cluster_size + self.eps) / (n + self.n_embed * self.eps) * n
            )
            embed_normalized = self.embed_avg / cluster_size.unsqueeze(0)
            self.embed.data.copy_(embed_normalized)

        diff = (quantize.detach() - input).pow(2).mean()
        quantize = input + (quantize - input).detach()

        return quantize, diff, embed_ind

    def embed_code(self, embed_id):
        return F.embedding(embed_id, self.embed.transpose(0, 1))


class ResBlock(nn.Module):
    def __init__(self, in_channel, channel):
        super().__init__()
        self.conv = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channel, channel, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel, in_channel, 1),
        )

    def forward(self, input):
        out = self.conv(input)
        out += input

        return out


class Encoder(nn.Module):
    def __init__(self, in_channel, channel, n_res_block, n_res_channel, stride):
        super().__init__()

        if stride == 4:
            blocks = [
                nn.Conv2d(in_channel, channel // 2, 4, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // 2, channel, 4, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel, channel, 3, padding=1),
            ]

        elif stride == 2:
            blocks = [
                nn.Conv2d(in_channel, channel // 2, 4, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // 2, channel, 3, padding=1),
            ]

        for i in range(n_res_block):
            blocks.append(ResBlock(channel, n_res_channel))

        blocks.append(nn.ReLU(inplace=True))

        self.blocks = nn.Sequential(*blocks)

    def forward(self, input):
        return self.blocks(input)


class Decoder(nn.Module):
    def __init__(
            self, in_channel, out_channel, channel, n_res_block, n_res_channel, stride
    ):
        super().__init__()

        blocks = [nn.Conv2d(in_channel, channel, 3, padding=1)]

        for i in range(n_res_block):
            blocks.append(ResBlock(channel, n_res_channel))

        blocks.append(nn.ReLU(inplace=True))

        if stride == 4:
            blocks.extend(
                [
                    nn.ConvTranspose2d(channel, channel // 2, 4, stride=2, padding=1),
                    nn.ReLU(inplace=True),
                    nn.ConvTranspose2d(
                        channel // 2, out_channel, 4, stride=2, padding=1
                    ),
                ]
            )

        elif stride == 2:
            blocks.append(
                nn.ConvTranspose2d(channel, out_channel, 4, stride=2, padding=1)
            )

        elif stride == 8:
            blocks.extend(
                [
                    nn.ConvTranspose2d(channel, channel // 4, 4, stride=2, padding=1),
                    nn.ReLU(inplace=True),
                    nn.ConvTranspose2d(channel // 4, channel // 2, 4, stride=2, padding=1),
                    nn.ReLU(inplace=True),
                    nn.ConvTranspose2d(channel // 2, out_channel, 4, stride=2, padding=1),
                ]
            )
        self.blocks = nn.Sequential(*blocks)

    def my_cosine_sync(self, x1, x2, x3, kernel_size=3, sigma=1.5):
        nb, ch, row, col = x1.size()
        cosine_1 = F.cosine_similarity(x1, x2, dim=1).unsqueeze(1)
        cosine_2 = F.cosine_similarity(x1, x3, dim=1).unsqueeze(1)
        cosine = cosine_1 + cosine_2
        filters = self._gaussian_filter(1, kernel_size, sigma).to(cosine.device)
        attenMap = F.conv2d(cosine, filters, padding=1)
        attenMap = attenMap.expand(nb, ch, row, col).contiguous()
        x2 = x2 * attenMap
        x3 = x3 * attenMap
        return x1, x2, x3

    def forward(self, input):
        return self.blocks(input)


class ResNet_VQ2(nn.Module):

    def __init__(self, block, layers, src_ch, tar_ch, BNmode, embed_dim=64, n_embed=512):
        super(ResNet_VQ2, self).__init__()
        kernels = [64, 128, 256, 512]
        self.src_ch = src_ch
        self.inplanes = kernels[0]
        self.stem_conv = nn.Conv2d(3, kernels[0], kernel_size=7,
                                   stride=2, padding=3, bias=False)
        print("####1 BNmode => ", BNmode)
        if BNmode == 'BN':
            self.stem_norm = nn.BatchNorm2d(kernels[0])
        elif BNmode == 'IN':
            self.stem_norm = nn.InstanceNorm2d(kernels[0])
        elif BNmode == 'GN':
            self.stem_norm = nn.GroupNorm(32, kernels[0])
        # Adding AdaIN, combined BN and IN
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(block, kernels[0], layers[0], stride=1, BNmode=BNmode)
        self.layer2 = self._make_layer(block, kernels[1], layers[1], stride=2, BNmode=BNmode)
        self.layer3 = self._make_layer(block, kernels[2], layers[2], stride=2, BNmode=BNmode)
        self.layer4 = self._make_layer(block, kernels[3], layers[3], stride=2, BNmode=BNmode)
        self.quantize_conv_t = nn.Conv2d(kernels[3], embed_dim, 1)
        self.quantize_t = Quantize(embed_dim, n_embed)
        self.dec_t = Decoder(embed_dim, embed_dim, 128, 2, 32, stride=2)
        self.quantize_conv_b = nn.Conv2d(embed_dim + kernels[2], embed_dim, 1)
        self.quantize_b = Quantize(embed_dim, n_embed)
        self.upsample_t = nn.ConvTranspose2d(embed_dim, embed_dim, 4, stride=2, padding=1)
        self.channel_dim = nn.Conv2d(embed_dim + embed_dim, kernels[2], kernel_size=1, stride=1)
        # self.dec = Decoder(embed_dim + embed_dim, in_channel, channel, n_res_block, n_res_channel, stride=4)

        # L components
        self.deconvL1 = upconv(kernels[2], kernels[1], ratio="x2")
        self.inplanes = kernels[1]
        self.upResL1 = self._make_layer(block, kernels[1], layers[1], BNmode=BNmode)

        self.deconvL2 = upconv(kernels[1], kernels[0], ratio="x2")
        self.inplanes = kernels[0]
        self.upResL2 = self._make_layer(block, kernels[0], layers[0], BNmode=BNmode)

        self.deconvL3 = upconv(kernels[0], kernels[0], ratio="x2")
        self.predL = nn.Conv2d(kernels[0], tar_ch[0], kernel_size=3, stride=1, padding=1, bias=False)

        # a components
        self.deconva1 = upconv(kernels[2], kernels[1], ratio="x2")
        self.inplanes = kernels[1]
        self.upResa1 = self._make_layer(block, kernels[1], layers[1], BNmode=BNmode)
        self.deconva2 = upconv(kernels[1], kernels[0], ratio="x2")
        self.inplanes = kernels[0]
        self.upResa2 = self._make_layer(block, kernels[0], layers[0], BNmode=BNmode)
        self.deconva3 = upconv(kernels[0], kernels[0], ratio="x2")
        self.preda = nn.Conv2d(kernels[0], tar_ch[1], kernel_size=3, stride=1, padding=1, bias=False)

        # b components
        self.deconvb1 = upconv(kernels[2], kernels[1], ratio="x2")
        self.inplanes = kernels[1]
        self.upResb1 = self._make_layer(block, kernels[1], layers[1], BNmode=BNmode)
        self.deconvb2 = upconv(kernels[1], kernels[0], ratio="x2")
        self.inplanes = kernels[0]
        self.upResb2 = self._make_layer(block, kernels[0], layers[0], BNmode=BNmode)
        self.deconvb3 = upconv(kernels[0], kernels[0], ratio="x2")
        self.predb = nn.Conv2d(kernels[0], tar_ch[2], kernel_size=3, stride=1, padding=1, bias=False)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1, BNmode='BN'):
        downsample = None
        print("####2 BNmode => ", BNmode)
        if stride != 1 or self.inplanes != planes * block.expansion:
            bnl = nn.BatchNorm2d(planes * block.expansion)
            if BNmode == 'IN':
                bnl = nn.InstanceNorm2d(planes * block.expansion)
            if BNmode == 'GN':
                bnl = nn.GroupNorm(32, planes * block.expansion)
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                bnl,
            )
            # Adding AdaIN

        layers = []
        if BNmode == 'GN':
            layers.append(block(self.inplanes, planes, stride, downsample, BNmode=BNmode))
        else:
            layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            if BNmode == 'GN':
                layers.append(block(self.inplanes, planes, BNmode=BNmode))
            else:
                layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    @staticmethod
    def _gaussian_filter(channel, kernel_size, sigma=1.5):
        gauss = torch.Tensor(
            [math.exp(-(x - kernel_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(kernel_size)])
        gauss_1D = (gauss / gauss.sum()).unsqueeze(1)
        gauss_2D = gauss_1D.mm(gauss_1D.t()).float().unsqueeze(0).unsqueeze(0)
        return gauss_2D.expand(channel, 1, kernel_size, kernel_size).contiguous()

    @staticmethod
    def my_gaussian_filter(channels, kernel_size=15, sigma=3):
        kernel_size = kernel_size  # 15
        sigma = sigma

        x_cord = torch.arange(kernel_size)
        x_grid = x_cord.repeat(kernel_size).view(kernel_size, kernel_size)
        y_grid = x_grid.t()
        xy_grid = torch.stack([x_grid, y_grid], dim=-1)

        mean = (kernel_size - 1) / 2.
        variance = sigma ** 2

        gaussian_kernel = (1. / (2. * math.pi * variance)) * \
                          torch.exp(-torch.sum((xy_grid - mean) ** 2., dim=-1) / \
                                    (2 * variance))
        gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)
        gaussian_kernel = gaussian_kernel.repeat(channels, 1, 1, 1)

        return gaussian_kernel

    def my_cosine_sync(self, x1, x2, x3, kernel_size=3, sigma=1.5):
        nb, ch, row, col = x1.size()
        cosine_1 = F.cosine_similarity(x1, x2, dim=1).unsqueeze(1)
        cosine_2 = F.cosine_similarity(x1, x3, dim=1).unsqueeze(1)
        cosine = cosine_1 + cosine_2
        filters = self._gaussian_filter(1, kernel_size, sigma).to(cosine.device)
        attenMap = F.conv2d(cosine, filters, padding=1)
        attenMap = attenMap.expand(nb, ch, row, col).contiguous()
        x2 = x2 * attenMap
        x3 = x3 * attenMap
        return x1, x2, x3

    def _status_sync(self, x1, x2, kernel_size=3, sigma=1.5):
        nb, ch, row, col = x1.size()
        cosine = F.cosine_similarity(x1, x2, dim=1).unsqueeze(1)
        filters = self._gaussian_filter(1, kernel_size, sigma).to(cosine.device)
        # print("Cosine =>", cosine.size(), "Filter =>", filters.size())
        attenMap = F.conv2d(cosine, filters, padding=1)
        attenMap = attenMap.expand(nb, ch, row, col).contiguous()
        x1 = x1 * attenMap
        x2 = x2 * attenMap
        return x1, x2

    def forward(self, x):
        if self.src_ch == 1:
            x = torch.cat([x, x, x], dim=1)
        # forward
        x1 = self.stem_conv(x)
        x1 = self.stem_norm(x1)
        x2 = self.relu(x1)  # enc_b

        # Encoder
        x3 = self.layer1(x2)
        x4 = self.layer2(x3)
        x5 = self.layer3(x4)
        x6 = self.layer4(x5)  # enc_t

        quant_t = self.quantize_conv_t(x6).permute(0, 2, 3, 1)
        quant_t, diff_t, id_t = self.quantize_t(quant_t)
        quant_t = quant_t.permute(0, 3, 1, 2)
        diff_t = diff_t.unsqueeze(0)
        dec_t = self.dec_t(quant_t)
        enc_b = torch.cat([dec_t, x5], 1)
        quant_b = self.quantize_conv_b(enc_b).permute(0, 2, 3, 1)
        quant_b, diff_b, id_b = self.quantize_b(quant_b)
        quant_b = quant_b.permute(0, 3, 1, 2)
        diff_b = diff_b.unsqueeze(0)
        upsample_t = self.upsample_t(quant_t)
        quant = torch.cat([upsample_t, quant_b], 1)
        quant = self.channel_dim(quant)
        x7_L = self.deconvL1(quant)
        x7_a = self.deconva1(quant)
        x7_b = self.deconvb1(quant)

        # sync
        x7_L, x7_a, x7_b = self.my_cosine_sync(x7_L, x7_a, x7_b)
        x7_L = x7_L + x4
        x7_a = x7_a + x4
        x7_b = x7_b + x4

        ## stage 2
        x8_L = self.upResL1(x7_L)
        x8_L = self.deconvL2(x8_L)
        x8_a = self.upResa1(x7_a)
        x8_a = self.deconva2(x8_a)
        x8_b = self.upResb1(x7_b)
        x8_b = self.deconvb2(x8_b)
        # sync
        x8_L, x8_a, x8_b = self.my_cosine_sync(x8_L, x8_a, x8_b)

        x8_L = x8_L + x3
        x8_a = x8_a + x3
        x8_b = x8_b + x3

        ## stage 3
        x9_L = self.upResL2(x8_L)
        x9_L = self.deconvL3(x9_L)
        x9_a = self.upResa2(x8_a)
        x9_a = self.deconva3(x9_a)
        x9_b = self.upResb2(x8_b)
        x9_b = self.deconvb3(x9_b)
        # sync
        x9_L, x9_a, x9_b = self.my_cosine_sync(x9_L, x9_a, x9_b)

        ## Final stage
        x_final_L = self.predL(x9_L)
        x_final_a = self.preda(x9_a)
        x_final_b = self.predb(x9_b)

        return x_final_L, x_final_a, x_final_b, diff_t + diff_b


class VQVAE(nn.Module):
    def __init__(self, in_channel=3, channel=128, n_res_block=2, n_res_channel=32,
                 embed_dim=64, n_embed=512, decay=0.99):
        super().__init__()

        self.enc_b = Encoder(in_channel, channel, n_res_block, n_res_channel, stride=4)
        self.enc_t = Encoder(channel, channel, n_res_block, n_res_channel, stride=2)
        self.quantize_conv_t = nn.Conv2d(channel, embed_dim, 1)
        self.quantize_t = Quantize(embed_dim, n_embed)
        self.dec_t = Decoder(
            embed_dim, embed_dim, channel, n_res_block, n_res_channel, stride=2
        )
        self.quantize_conv_b = nn.Conv2d(embed_dim + channel, embed_dim, 1)
        self.quantize_b = Quantize(embed_dim, n_embed)
        self.upsample_t = nn.ConvTranspose2d(
            embed_dim, embed_dim, 4, stride=2, padding=1
        )
        self.dec = Decoder(
            embed_dim + embed_dim,
            in_channel,
            channel,
            n_res_block,
            n_res_channel,
            stride=4,
        )

    def forward(self, input):
        quant_t, quant_b, diff, _, _ = self.encode(input)
        dec = self.decode(quant_t, quant_b)

        return dec, diff

    def encode(self, input):
        enc_b = self.enc_b(input)
        enc_t = self.enc_t(enc_b)

        quant_t = self.quantize_conv_t(enc_t).permute(0, 2, 3, 1)
        quant_t, diff_t, id_t = self.quantize_t(quant_t)
        quant_t = quant_t.permute(0, 3, 1, 2)
        diff_t = diff_t.unsqueeze(0)

        dec_t = self.dec_t(quant_t)
        enc_b = torch.cat([dec_t, enc_b], 1)

        quant_b = self.quantize_conv_b(enc_b).permute(0, 2, 3, 1)
        quant_b, diff_b, id_b = self.quantize_b(quant_b)
        quant_b = quant_b.permute(0, 3, 1, 2)
        diff_b = diff_b.unsqueeze(0)

        return quant_t, quant_b, diff_t + diff_b, id_t, id_b

    def decode(self, quant_t, quant_b):
        upsample_t = self.upsample_t(quant_t)
        quant = torch.cat([upsample_t, quant_b], 1)
        dec = self.dec(quant)

        return dec

    def decode_code(self, code_t, code_b):
        quant_t = self.quantize_t.embed_code(code_t)
        quant_t = quant_t.permute(0, 3, 1, 2)
        quant_b = self.quantize_b.embed_code(code_b)
        quant_b = quant_b.permute(0, 3, 1, 2)

        dec = self.decode(quant_t, quant_b)

        return dec


class ResNet_VQ(nn.Module):
    def __init__(self, in_channel=3, channel=128, n_res_block=2, n_res_channel=32,
                 embed_dim=64, n_embed=512, decay=0.99):
        super().__init__()
        self.enc_b = Encoder(in_channel, channel, n_res_block, n_res_channel, stride=4)
        self.enc_t = Encoder(channel, channel, n_res_block, n_res_channel, stride=2)
        self.quantize_conv_t = nn.Conv2d(channel, embed_dim, 1)
        self.quantize_t = Quantize(embed_dim, n_embed)
        self.dec_t = Decoder(embed_dim, embed_dim, channel, n_res_block, n_res_channel, stride=2)
        self.quantize_conv_b = nn.Conv2d(embed_dim + channel, embed_dim, 1)
        self.quantize_b = Quantize(embed_dim, n_embed)
        self.upsample_t = nn.ConvTranspose2d(embed_dim, embed_dim, 4, stride=2, padding=1)
        # self.dec = Decoder(embed_dim + embed_dim, in_channel, channel, n_res_block, n_res_channel, stride=4)
        self.dec_1 = nn.Conv2d(embed_dim + embed_dim, channel, 3, padding=1)
        blocks = []
        for i in range(n_res_block):
            blocks.append(ResBlock(channel, n_res_channel))
        blocks.append(nn.ReLU(inplace=True))
        self.blocks = nn.Sequential(*blocks)
        self.relu = nn.ReLU(inplace=True)
        self.dec_2 = nn.ConvTranspose2d(channel, channel // 2, 4, stride=2, padding=1)
        self.dec_3 = nn.ConvTranspose2d(channel // 2, 3, 4, stride=2, padding=1)

    def forward(self, input):
        # forward
        quant_t, quant_b, diff, _, _ = self.encode(input)
        upsample_t = self.upsample_t(quant_t)
        quant = torch.cat([upsample_t, quant_b], 1)
        dec_1 = self.dec_1(quant)
        dec_blocks = self.blocks(dec_1)
        dec_2 = self.dec_2(dec_blocks)
        dec_3 = self.dec_3(dec_2)
        # dec = self.decode(quant_t, quant_b)
        return dec_3[:, :1, :, :], dec_3[:, 1:, :, :], diff

    def encode(self, input):
        enc_b = self.enc_b(input)
        enc_t = self.enc_t(enc_b)
        quant_t = self.quantize_conv_t(enc_t).permute(0, 2, 3, 1)
        quant_t, diff_t, id_t = self.quantize_t(quant_t)
        quant_t = quant_t.permute(0, 3, 1, 2)
        diff_t = diff_t.unsqueeze(0)

        dec_t = self.dec_t(quant_t)
        enc_b = torch.cat([dec_t, enc_b], 1)

        quant_b = self.quantize_conv_b(enc_b).permute(0, 2, 3, 1)
        quant_b, diff_b, id_b = self.quantize_b(quant_b)
        quant_b = quant_b.permute(0, 3, 1, 2)
        diff_b = diff_b.unsqueeze(0)

        return quant_t, quant_b, diff_t + diff_b, id_t, id_b

    def decode(self, quant_t, quant_b):
        upsample_t = self.upsample_t(quant_t)
        quant = torch.cat([upsample_t, quant_b], 1)
        dec = self.dec(quant)

        return dec

    def decode_code(self, code_t, code_b):
        quant_t = self.quantize_t.embed_code(code_t)
        quant_t = quant_t.permute(0, 3, 1, 2)
        quant_b = self.quantize_b.embed_code(code_b)
        quant_b = quant_b.permute(0, 3, 1, 2)

        dec = self.decode(quant_t, quant_b)

        return dec


class ResNet_VQ3(nn.Module):

    def __init__(self, block, layers, src_ch, tar_ch, BNmode,
                 in_channel=3, channel=128, n_res_block=2, n_res_channel=32, embed_dim=256, n_embed=512):
        super(ResNet_VQ3, self).__init__()

        self.enc_b = Encoder(in_channel, channel, n_res_block, n_res_channel, stride=4)
        self.enc_t = Encoder(channel, channel, n_res_block, n_res_channel, stride=2)
        self.quantize_conv_t = nn.Conv2d(channel, embed_dim, 1)
        self.quantize_t = Quantize(embed_dim, n_embed)
        self.dec_t = Decoder(
            embed_dim, embed_dim, channel, n_res_block, n_res_channel, stride=2
        )
        self.quantize_conv_b = nn.Conv2d(embed_dim + channel, embed_dim, 1)
        self.quantize_b = Quantize(embed_dim, n_embed)
        self.upsample_t = nn.ConvTranspose2d(
            embed_dim, embed_dim, 4, stride=2, padding=1
        )
        self.dec = Decoder(
            embed_dim + embed_dim,
            in_channel,
            channel,
            n_res_block,
            n_res_channel,
            stride=4,
        )

        kernels = [64, 128, 256, 512]

        self.src_ch = src_ch
        self.inplanes = kernels[0]
        self.stem_conv = nn.Conv2d(3, kernels[0], kernel_size=7,
                                   stride=2, padding=3, bias=False)
        print("####1 BNmode => ", BNmode)
        if BNmode == 'BN':
            self.stem_norm = nn.BatchNorm2d(kernels[0])
        elif BNmode == 'IN':
            self.stem_norm = nn.InstanceNorm2d(kernels[0])
        elif BNmode == 'GN':
            self.stem_norm = nn.GroupNorm(32, kernels[0])
        # Adding AdaIN, combined BN and IN
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(block, kernels[0], layers[0], stride=1, BNmode=BNmode)
        self.layer2 = self._make_layer(block, kernels[1], layers[1], stride=2, BNmode=BNmode)
        self.layer3 = self._make_layer(block, kernels[2], layers[2], stride=2, BNmode=BNmode)
        self.inplanes = kernels[2] + kernels[2]
        self.layer4 = self._make_layer(block, kernels[3], layers[3], stride=2, BNmode=BNmode)

        # Red components
        self.deconvL1 = upconv(kernels[3], kernels[2], ratio="x2")
        self.inplanes = kernels[2]
        self.upResL1 = self._make_layer(block, kernels[2], layers[2], BNmode=BNmode)
        self.deconvL2 = upconv(kernels[2], kernels[1], ratio="x2")
        self.inplanes = kernels[1]
        self.upResL2 = self._make_layer(block, kernels[1], layers[1], BNmode=BNmode)
        self.deconvL3 = upconv(kernels[1], kernels[0], ratio="x2")
        self.inplanes = kernels[0]
        self.upResL3 = self._make_layer(block, kernels[0], layers[0], BNmode=BNmode)
        self.deconvL4 = upconv(kernels[0], kernels[0], ratio="x2")
        self.predL = nn.Conv2d(kernels[0], tar_ch[0], kernel_size=3, stride=1, padding=1, bias=False)

        # a components
        self.deconva1 = upconv(kernels[3], kernels[2], ratio="x2")
        self.inplanes = kernels[2]
        self.upResa1 = self._make_layer(block, kernels[2], layers[2], BNmode=BNmode)
        self.deconva2 = upconv(kernels[2], kernels[1], ratio="x2")
        self.inplanes = kernels[1]
        self.upResa2 = self._make_layer(block, kernels[1], layers[1], BNmode=BNmode)
        self.deconva3 = upconv(kernels[1], kernels[0], ratio="x2")
        self.inplanes = kernels[0]
        self.upResa3 = self._make_layer(block, kernels[0], layers[0], BNmode=BNmode)
        self.deconva4 = upconv(kernels[0], kernels[0], ratio="x2")
        self.preda = nn.Conv2d(kernels[0], tar_ch[1], kernel_size=3, stride=1, padding=1, bias=False)

        # b components
        self.deconvb1 = upconv(kernels[3], kernels[2], ratio="x2")
        self.inplanes = kernels[2]
        self.upResb1 = self._make_layer(block, kernels[2], layers[2], BNmode=BNmode)
        self.deconvb2 = upconv(kernels[2], kernels[1], ratio="x2")
        self.inplanes = kernels[1]
        self.upResb2 = self._make_layer(block, kernels[1], layers[1], BNmode=BNmode)
        self.deconvb3 = upconv(kernels[1], kernels[0], ratio="x2")
        self.inplanes = kernels[0]
        self.upResb3 = self._make_layer(block, kernels[0], layers[0], BNmode=BNmode)
        self.deconvb4 = upconv(kernels[0], kernels[0], ratio="x2")
        self.predb = nn.Conv2d(kernels[0], tar_ch[2], kernel_size=3, stride=1, padding=1, bias=False)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def encode(self, input):
        enc_b = self.enc_b(input)
        enc_t = self.enc_t(enc_b)
        quant_t = self.quantize_conv_t(enc_t).permute(0, 2, 3, 1)
        quant_t, diff_t, id_t = self.quantize_t(quant_t)
        quant_t = quant_t.permute(0, 3, 1, 2)
        diff_t = diff_t.unsqueeze(0)

        dec_t = self.dec_t(quant_t)
        enc_b = torch.cat([dec_t, enc_b], 1)

        quant_b = self.quantize_conv_b(enc_b).permute(0, 2, 3, 1)
        quant_b, diff_b, id_b = self.quantize_b(quant_b)
        quant_b = quant_b.permute(0, 3, 1, 2)
        diff_b = diff_b.unsqueeze(0)
        # enc_b ?
        return enc_t, quant_t, quant_b, diff_t + diff_b, id_t, id_b

    def decode(self, quant_t, quant_b):
        upsample_t = self.upsample_t(quant_t)
        quant = torch.cat([upsample_t, quant_b], 1)
        dec = self.dec(quant)

        return dec

    def _make_layer(self, block, planes, blocks, stride=1, BNmode='BN'):
        downsample = None
        print("####2 BNmode => ", BNmode)
        if stride != 1 or self.inplanes != planes * block.expansion:
            bnl = nn.BatchNorm2d(planes * block.expansion)
            if BNmode == 'IN':
                bnl = nn.InstanceNorm2d(planes * block.expansion)
            if BNmode == 'GN':
                bnl = nn.GroupNorm(32, planes * block.expansion)
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                bnl,
            )
            # Adding AdaIN

        layers = []
        if BNmode == 'GN':
            layers.append(block(self.inplanes, planes, stride, downsample, BNmode=BNmode))
        else:
            layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            if BNmode == 'GN':
                layers.append(block(self.inplanes, planes, BNmode=BNmode))
            else:
                layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    @staticmethod
    def _gaussian_filter(channel, kernel_size, sigma=1.5):
        gauss = torch.Tensor(
            [math.exp(-(x - kernel_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(kernel_size)])
        gauss_1D = (gauss / gauss.sum()).unsqueeze(1)
        gauss_2D = gauss_1D.mm(gauss_1D.t()).float().unsqueeze(0).unsqueeze(0)
        return gauss_2D.expand(channel, 1, kernel_size, kernel_size).contiguous()

    def my_cosine_sync(self, x1, x2, x3, kernel_size=3, sigma=1.5):
        nb, ch, row, col = x1.size()
        cosine_1 = F.cosine_similarity(x1, x2, dim=1).unsqueeze(1)
        cosine_2 = F.cosine_similarity(x1, x3, dim=1).unsqueeze(1)
        cosine = cosine_1 + cosine_2
        filters = self._gaussian_filter(1, kernel_size, sigma).to(cosine.device)
        attenMap = F.conv2d(cosine, filters, padding=1)
        attenMap = attenMap.expand(nb, ch, row, col).contiguous()
        x2 = x2 * attenMap
        x3 = x3 * attenMap
        return x1, x2, x3

    def _status_sync(self, x1, x2, kernel_size=3, sigma=1.5):
        nb, ch, row, col = x1.size()
        cosine = F.cosine_similarity(x1, x2, dim=1).unsqueeze(1)
        filters = self._gaussian_filter(1, kernel_size, sigma).to(cosine.device)
        # print("Cosine =>", cosine.size(), "Filter =>", filters.size())
        attenMap = F.conv2d(cosine, filters, padding=1)
        attenMap = attenMap.expand(nb, ch, row, col).contiguous()
        x1 = x1 * attenMap
        x2 = x2 * attenMap
        return x1, x2

    def forward(self, x):
        if self.src_ch == 1:
            x = torch.cat([x, x, x], dim=1)
        # forward

        enc_t, quant_t, quant_b, diff, _, _ = self.encode(x)
        dec = self.decode(quant_t, quant_b)

        x1 = self.stem_conv(x)
        x1 = self.stem_norm(x1)
        x2 = self.relu(x1)

        # Encoder
        x3 = self.layer1(x2)
        x4 = self.layer2(x3)
        x5 = self.layer3(x4)
        x5_cat = torch.cat([x5, quant_t], 1)
        x6 = self.layer4(x5_cat)

        ## stage 1
        x7_L = self.deconvL1(x6)
        x7_a = self.deconva1(x6)
        x7_b = self.deconvb1(x6)
        # sync
        x7_L, x7_a, x7_b = self.my_cosine_sync(x7_L, x7_a, x7_b)

        x7_L = x7_L + x5
        x7_a = x7_a + x5
        x7_b = x7_b + x5

        ## stage 2
        x8_L = self.upResL1(x7_L)
        x8_L = self.deconvL2(x8_L)
        x8_a = self.upResa1(x7_a)
        x8_a = self.deconva2(x8_a)
        x8_b = self.upResb1(x7_b)
        x8_b = self.deconvb2(x8_b)
        # sync
        x8_L, x8_a, x8_b = self.my_cosine_sync(x8_L, x8_a, x8_b)

        x8_L = x8_L + x4
        x8_a = x8_a + x4
        x8_b = x8_b + x4

        ## stage 3
        x9_L = self.upResL2(x8_L)
        x9_L = self.deconvL3(x9_L)
        x9_a = self.upResa2(x8_a)
        x9_a = self.deconva3(x9_a)
        x9_b = self.upResb2(x8_b)
        x9_b = self.deconvb3(x9_b)
        # sync
        x9_L, x9_a, x9_b = self.my_cosine_sync(x9_L, x9_a, x9_b)

        x9_L = x9_L + x3
        x9_a = x9_a + x3
        x9_b = x9_b + x3

        ## stage 4
        x10_L = self.upResL3(x9_L)
        x10_L = self.deconvL4(x10_L)
        x10_a = self.upResa3(x9_a)
        x10_a = self.deconva4(x10_a)
        x10_b = self.upResb3(x9_b)
        x10_b = self.deconvb4(x10_b)
        # sync
        x10_L, x10_a, x10_b = self.my_cosine_sync(x10_L, x10_a, x10_b)

        ## Final stage
        x_final_L = self.predL(x10_L)
        x_final_a = self.preda(x10_a)
        x_final_b = self.predb(x10_b)

        return x_final_L, x_final_a, x_final_b, dec, diff


class ResNet_pix2pixHD(nn.Module):

    def __init__(self, block, layers, src_ch, tar_ch, BNmode,
                 in_channel=3, channel=128, n_res_block=2, n_res_channel=32, embed_dim=256, n_embed=512):
        super(ResNet_pix2pixHD, self).__init__()

        self.enc_b = Encoder(in_channel, channel, n_res_block, n_res_channel, stride=4)
        self.enc_t = Encoder(channel, channel, n_res_block, n_res_channel, stride=2)
        self.quantize_conv_t = nn.Conv2d(channel, embed_dim, 1)
        self.quantize_t = Quantize(embed_dim, n_embed)
        self.dec_t = Decoder(
            embed_dim, embed_dim, channel, n_res_block, n_res_channel, stride=2
        )
        self.quantize_conv_b = nn.Conv2d(embed_dim + channel, embed_dim, 1)
        self.quantize_b = Quantize(embed_dim, n_embed)
        self.upsample_t = nn.ConvTranspose2d(
            embed_dim, embed_dim, 4, stride=2, padding=1
        )
        self.dec = Decoder(
            embed_dim + embed_dim,
            in_channel,
            channel,
            n_res_block,
            n_res_channel,
            stride=4,
        )

        kernels = [64, 128, 256, 512]

        self.src_ch = src_ch
        self.inplanes = kernels[0]
        self.stem_conv = nn.Conv2d(3, kernels[0], kernel_size=7,
                                   stride=2, padding=3, bias=False)
        print("####1 BNmode => ", BNmode)
        if BNmode == 'BN':
            self.stem_norm = nn.BatchNorm2d(kernels[0])
        elif BNmode == 'IN':
            self.stem_norm = nn.InstanceNorm2d(kernels[0])
        elif BNmode == 'GN':
            self.stem_norm = nn.GroupNorm(32, kernels[0])
        # Adding AdaIN, combined BN and IN
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(block, kernels[0], layers[0], stride=1, BNmode=BNmode)
        self.layer2 = self._make_layer(block, kernels[1], layers[1], stride=2, BNmode=BNmode)
        self.layer3 = self._make_layer(block, kernels[2], layers[2], stride=2, BNmode=BNmode)
        self.inplanes = kernels[2] + kernels[2]
        self.layer4 = self._make_layer(block, kernels[3], layers[3], stride=2, BNmode=BNmode)

        # Red components
        self.deconvL1 = upconv(kernels[3], kernels[2], ratio="x2")
        self.inplanes = kernels[2]
        self.upResL1 = self._make_layer(block, kernels[2], layers[2], BNmode=BNmode)
        self.deconvL2 = upconv(kernels[2], kernels[1], ratio="x2")
        self.inplanes = kernels[1]
        self.upResL2 = self._make_layer(block, kernels[1], layers[1], BNmode=BNmode)
        self.deconvL3 = upconv(kernels[1], kernels[0], ratio="x2")
        self.inplanes = kernels[0]
        self.upResL3 = self._make_layer(block, kernels[0], layers[0], BNmode=BNmode)
        self.deconvL4 = upconv(kernels[0], kernels[0], ratio="x2")
        self.predL = nn.Conv2d(kernels[0], tar_ch[0], kernel_size=3, stride=1, padding=1, bias=False)

        # a components
        self.deconva1 = upconv(kernels[3], kernels[2], ratio="x2")
        self.inplanes = kernels[2]
        self.upResa1 = self._make_layer(block, kernels[2], layers[2], BNmode=BNmode)
        self.deconva2 = upconv(kernels[2], kernels[1], ratio="x2")
        self.inplanes = kernels[1]
        self.upResa2 = self._make_layer(block, kernels[1], layers[1], BNmode=BNmode)
        self.deconva3 = upconv(kernels[1], kernels[0], ratio="x2")
        self.inplanes = kernels[0]
        self.upResa3 = self._make_layer(block, kernels[0], layers[0], BNmode=BNmode)
        self.deconva4 = upconv(kernels[0], kernels[0], ratio="x2")
        self.preda = nn.Conv2d(kernels[0], tar_ch[1], kernel_size=3, stride=1, padding=1, bias=False)

        # b components
        self.deconvb1 = upconv(kernels[3], kernels[2], ratio="x2")
        self.inplanes = kernels[2]
        self.upResb1 = self._make_layer(block, kernels[2], layers[2], BNmode=BNmode)
        self.deconvb2 = upconv(kernels[2], kernels[1], ratio="x2")
        self.inplanes = kernels[1]
        self.upResb2 = self._make_layer(block, kernels[1], layers[1], BNmode=BNmode)
        self.deconvb3 = upconv(kernels[1], kernels[0], ratio="x2")
        self.inplanes = kernels[0]
        self.upResb3 = self._make_layer(block, kernels[0], layers[0], BNmode=BNmode)
        self.deconvb4 = upconv(kernels[0], kernels[0], ratio="x2")
        self.predb = nn.Conv2d(kernels[0], tar_ch[2], kernel_size=3, stride=1, padding=1, bias=False)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def encode(self, input):
        enc_b = self.enc_b(input)
        enc_t = self.enc_t(enc_b)
        quant_t = self.quantize_conv_t(enc_t).permute(0, 2, 3, 1)
        quant_t, diff_t, id_t = self.quantize_t(quant_t)
        quant_t = quant_t.permute(0, 3, 1, 2)
        diff_t = diff_t.unsqueeze(0)

        dec_t = self.dec_t(quant_t)
        enc_b = torch.cat([dec_t, enc_b], 1)

        quant_b = self.quantize_conv_b(enc_b).permute(0, 2, 3, 1)
        quant_b, diff_b, id_b = self.quantize_b(quant_b)
        quant_b = quant_b.permute(0, 3, 1, 2)
        diff_b = diff_b.unsqueeze(0)
        # enc_b ?
        return enc_t, quant_t, quant_b, diff_t + diff_b, id_t, id_b

    def decode(self, quant_t, quant_b):
        upsample_t = self.upsample_t(quant_t)
        quant = torch.cat([upsample_t, quant_b], 1)
        dec = self.dec(quant)

        return dec

    def _make_layer(self, block, planes, blocks, stride=1, BNmode='BN'):
        downsample = None
        print("####2 BNmode => ", BNmode)
        if stride != 1 or self.inplanes != planes * block.expansion:
            bnl = nn.BatchNorm2d(planes * block.expansion)
            if BNmode == 'IN':
                bnl = nn.InstanceNorm2d(planes * block.expansion)
            if BNmode == 'GN':
                bnl = nn.GroupNorm(32, planes * block.expansion)
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                bnl,
            )
            # Adding AdaIN

        layers = []
        if BNmode == 'GN':
            layers.append(block(self.inplanes, planes, stride, downsample, BNmode=BNmode))
        else:
            layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            if BNmode == 'GN':
                layers.append(block(self.inplanes, planes, BNmode=BNmode))
            else:
                layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    @staticmethod
    def _gaussian_filter(channel, kernel_size, sigma=1.5):
        gauss = torch.Tensor(
            [math.exp(-(x - kernel_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(kernel_size)])
        gauss_1D = (gauss / gauss.sum()).unsqueeze(1)
        gauss_2D = gauss_1D.mm(gauss_1D.t()).float().unsqueeze(0).unsqueeze(0)
        return gauss_2D.expand(channel, 1, kernel_size, kernel_size).contiguous()

    def my_cosine_sync(self, x1, x2, x3, kernel_size=3, sigma=1.5):
        nb, ch, row, col = x1.size()
        cosine_1 = F.cosine_similarity(x1, x2, dim=1).unsqueeze(1)
        cosine_2 = F.cosine_similarity(x1, x3, dim=1).unsqueeze(1)
        cosine = cosine_1 + cosine_2
        filters = self._gaussian_filter(1, kernel_size, sigma).to(cosine.device)
        attenMap = F.conv2d(cosine, filters, padding=1)
        attenMap = attenMap.expand(nb, ch, row, col).contiguous()
        x2 = x2 * attenMap
        x3 = x3 * attenMap
        return x1, x2, x3

    def _status_sync(self, x1, x2, kernel_size=3, sigma=1.5):
        nb, ch, row, col = x1.size()
        cosine = F.cosine_similarity(x1, x2, dim=1).unsqueeze(1)
        filters = self._gaussian_filter(1, kernel_size, sigma).to(cosine.device)
        # print("Cosine =>", cosine.size(), "Filter =>", filters.size())
        attenMap = F.conv2d(cosine, filters, padding=1)
        attenMap = attenMap.expand(nb, ch, row, col).contiguous()
        x1 = x1 * attenMap
        x2 = x2 * attenMap
        return x1, x2

    def forward(self, x):
        if self.src_ch == 1:
            x = torch.cat([x, x, x], dim=1)
        # forward

        enc_t, quant_t, quant_b, diff, _, _ = self.encode(x)
        dec = self.decode(quant_t, quant_b)

        x1 = self.stem_conv(x)
        x1 = self.stem_norm(x1)
        x2 = self.relu(x1)

        # Encoder
        x3 = self.layer1(x2)
        x4 = self.layer2(x3)
        x5 = self.layer3(x4)
        x5_cat = torch.cat([x5, quant_t], 1)
        x6 = self.layer4(x5_cat)

        ## stage 1
        x7_L = self.deconvL1(x6)
        x7_a = self.deconva1(x6)
        x7_b = self.deconvb1(x6)
        # sync
        x7_L, x7_a, x7_b = self.my_cosine_sync(x7_L, x7_a, x7_b)

        x7_L = x7_L + x5
        x7_a = x7_a + x5
        x7_b = x7_b + x5

        ## stage 2
        x8_L = self.upResL1(x7_L)
        x8_L = self.deconvL2(x8_L)
        x8_a = self.upResa1(x7_a)
        x8_a = self.deconva2(x8_a)
        x8_b = self.upResb1(x7_b)
        x8_b = self.deconvb2(x8_b)
        # sync
        x8_L, x8_a, x8_b = self.my_cosine_sync(x8_L, x8_a, x8_b)

        x8_L = x8_L + x4
        x8_a = x8_a + x4
        x8_b = x8_b + x4

        ## stage 3
        x9_L = self.upResL2(x8_L)
        x9_L = self.deconvL3(x9_L)
        x9_a = self.upResa2(x8_a)
        x9_a = self.deconva3(x9_a)
        x9_b = self.upResb2(x8_b)
        x9_b = self.deconvb3(x9_b)
        # sync
        x9_L, x9_a, x9_b = self.my_cosine_sync(x9_L, x9_a, x9_b)

        x9_L = x9_L + x3
        x9_a = x9_a + x3
        x9_b = x9_b + x3

        ## stage 4
        x10_L = self.upResL3(x9_L)
        x10_L = self.deconvL4(x10_L)
        x10_a = self.upResa3(x9_a)
        x10_a = self.deconva4(x10_a)
        x10_b = self.upResb3(x9_b)
        x10_b = self.deconvb4(x10_b)
        # sync
        x10_L, x10_a, x10_b = self.my_cosine_sync(x10_L, x10_a, x10_b)

        ## Final stage
        x_final_L = self.predL(x10_L)
        x_final_a = self.preda(x10_a)
        x_final_b = self.predb(x10_b)

        return x_final_L, x_final_a, x_final_b, dec, diff


def get_gaussian_kernel(k=3, mu=0, sigma=1, normalize=True):
    # compute 1 dimension gaussian
    gaussian_1D = np.linspace(-1, 1, k)
    # compute a grid distance from center
    x, y = np.meshgrid(gaussian_1D, gaussian_1D)
    distance = (x ** 2 + y ** 2) ** 0.5

    # compute the 2 dimension gaussian
    gaussian_2D = np.exp(-(distance - mu) ** 2 / (2 * sigma ** 2))
    gaussian_2D = gaussian_2D / (2 * np.pi * sigma ** 2)

    # normalize part (mathematically)
    if normalize:
        gaussian_2D = gaussian_2D / np.sum(gaussian_2D)
    return gaussian_2D


def get_sobel_kernel(k=3):
    # get range
    range = np.linspace(-(k // 2), k // 2, k)
    # compute a grid the numerator and the axis-distances
    x, y = np.meshgrid(range, range)
    sobel_2D_numerator = x
    sobel_2D_denominator = (x ** 2 + y ** 2)
    sobel_2D_denominator[:, k // 2] = 1  # avoid division by zero
    sobel_2D = sobel_2D_numerator / sobel_2D_denominator
    return sobel_2D


def get_thin_kernels(start=0, end=360, step=45):
    k_thin = 3  # actual size of the directional kernel
    # increase for a while to avoid interpolation when rotating
    k_increased = k_thin + 2

    # get 0° angle directional kernel
    thin_kernel_0 = np.zeros((k_increased, k_increased))
    thin_kernel_0[k_increased // 2, k_increased // 2] = 1
    thin_kernel_0[k_increased // 2, k_increased // 2 + 1:] = -1

    # rotate the 0° angle directional kernel to get the other ones
    thin_kernels = []
    for angle in range(start, end, step):
        (h, w) = thin_kernel_0.shape
        # get the center to not rotate around the (0, 0) coord point
        center = (w // 2, h // 2)
        # apply rotation
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1)
        kernel_angle_increased = cv2.warpAffine(thin_kernel_0, rotation_matrix, (w, h), cv2.INTER_NEAREST)

        # get the k=3 kerne
        kernel_angle = kernel_angle_increased[1:-1, 1:-1]
        is_diag = (abs(kernel_angle) == 1)  # because of the interpolation
        kernel_angle = kernel_angle * is_diag  # because of the interpolation
        thin_kernels.append(kernel_angle)
    return thin_kernels


class CannyFilter(nn.Module):
    def __init__(self,
                 k_gaussian=3,
                 mu=0,
                 sigma=1,
                 k_sobel=3,
                 use_cuda=False):
        super(CannyFilter, self).__init__()
        # device
        self.device = 'cuda:0' if use_cuda else 'cpu'

        # gaussian

        gaussian_2D = get_gaussian_kernel(k_gaussian, mu, sigma)
        self.gaussian_filter = nn.Conv2d(in_channels=1,
                                         out_channels=1,
                                         kernel_size=k_gaussian,
                                         padding=k_gaussian // 2,
                                         bias=False)

        self.gaussian_filter.weight.data.copy_(torch.from_numpy(gaussian_2D))

        # self.gaussian_filter.weight[:] = torch.from_numpy(gaussian_2D)
        # sobel

        sobel_2D = get_sobel_kernel(k_sobel)
        self.sobel_filter_x = nn.Conv2d(in_channels=1,
                                        out_channels=1,
                                        kernel_size=k_sobel,
                                        padding=k_sobel // 2,
                                        bias=False)
        self.sobel_filter_x.weight.data.copy_(torch.from_numpy(sobel_2D))

        # self.sobel_filter_x.weight[:] = torch.from_numpy(sobel_2D)

        self.sobel_filter_y = nn.Conv2d(in_channels=1,
                                        out_channels=1,
                                        kernel_size=k_sobel,
                                        padding=k_sobel // 2,
                                        bias=False)
        self.sobel_filter_y.weight.data.copy_(torch.from_numpy(sobel_2D.T))

        # self.sobel_filter_y.weight[:] = torch.from_numpy(sobel_2D.T)

        # thin

        thin_kernels = get_thin_kernels()
        directional_kernels = np.stack(thin_kernels)

        self.directional_filter = nn.Conv2d(in_channels=1,
                                            out_channels=8,
                                            kernel_size=thin_kernels[0].shape,
                                            padding=thin_kernels[0].shape[-1] // 2,
                                            bias=False)
        self.directional_filter.weight[:, 0].data.copy_(torch.from_numpy(directional_kernels))

        # hysteresis

        hysteresis = np.ones((3, 3)) + 0.25
        self.hysteresis = nn.Conv2d(in_channels=1,
                                    out_channels=1,
                                    kernel_size=3,
                                    padding=1,
                                    bias=False)
        self.hysteresis.weight.data.copy_(torch.from_numpy(hysteresis))

    def forward(self, img, low_threshold=None, high_threshold=None, hysteresis=False):
        # set the setps tensors
        B, C, H, W = img.shape
        blurred = torch.zeros((B, C, H, W)).to(self.device).detach()
        grad_x = torch.zeros((B, 1, H, W)).to(self.device).detach()
        grad_y = torch.zeros((B, 1, H, W)).to(self.device).detach()
        grad_magnitude = torch.zeros((B, 1, H, W)).to(self.device).detach()
        grad_orientation = torch.zeros((B, 1, H, W)).to(self.device).detach()

        # gaussian

        for c in range(C):
            blurred[:, c:c + 1] = self.gaussian_filter(img[:, c:c + 1])

            grad_x = grad_x + self.sobel_filter_x(blurred[:, c:c + 1])
            grad_y = grad_y + self.sobel_filter_y(blurred[:, c:c + 1])

        # thick edges

        grad_x, grad_y = grad_x / C, grad_y / C
        grad_magnitude = (grad_x ** 2 + grad_y ** 2) ** 0.5
        grad_orientation = torch.atan(grad_y / grad_x)
        grad_orientation = grad_orientation * (360 / np.pi) + 180  # convert to degree
        grad_orientation = torch.round(grad_orientation / 45) * 45  # keep a split by 45

        # thin edges

        directional = self.directional_filter(grad_magnitude)
        # get indices of positive and negative directions
        positive_idx = (grad_orientation / 45) % 8
        negative_idx = ((grad_orientation / 45) + 4) % 8
        thin_edges = grad_magnitude.clone()
        # non maximum suppression direction by direction
        for pos_i in range(4):
            neg_i = pos_i + 4
            # get the oriented grad for the angle
            is_oriented_i = (positive_idx == pos_i) * 1
            is_oriented_i = is_oriented_i + (positive_idx == neg_i) * 1
            pos_directional = directional[:, pos_i]
            neg_directional = directional[:, neg_i]
            selected_direction = torch.stack([pos_directional, neg_directional])

            # get the local maximum pixels for the angle
            is_max = selected_direction.min(dim=0)[0] > 0.0
            is_max = torch.unsqueeze(is_max, dim=1)

            # apply non maximum suppression
            to_remove = (is_max == 0) * 1 * (is_oriented_i) > 0
            thin_edges[to_remove] = 0.0

        # thresholds

        if low_threshold is not None:
            low = thin_edges > low_threshold

            if high_threshold is not None:
                high = thin_edges > high_threshold
                # get black/gray/white only
                thin_edges = low * 0.5 + high * 0.5

                if hysteresis:
                    # get weaks and check if they are high or not
                    weak = (thin_edges == 0.5) * 1
                    weak_is_high = (self.hysteresis(thin_edges) > 1) * weak
                    thin_edges = high * 1 + weak_is_high * 1
            else:
                thin_edges = low * 1

        return blurred.detach(), grad_x.detach(), grad_y.detach(), grad_magnitude.detach(), grad_orientation.detach(), thin_edges.detach()


class ResNet_pix2pixHD_Cannyedge(nn.Module):

    def __init__(self, block, layers, src_ch, tar_ch, BNmode,
                 in_channel=3, channel=128, n_res_block=2, n_res_channel=32, embed_dim=256, n_embed=512):
        super(ResNet_pix2pixHD_Cannyedge, self).__init__()

        self.edge_detection = CannyFilter(use_cuda=True)
        self.enc_b = Encoder(in_channel, channel, n_res_block, n_res_channel, stride=4)
        self.enc_t = Encoder(channel, channel, n_res_block, n_res_channel, stride=2)
        self.quantize_conv_t = nn.Conv2d(channel, embed_dim, 1)
        self.quantize_t = Quantize(embed_dim, n_embed)
        self.dec_t = Decoder(
            embed_dim, embed_dim, channel, n_res_block, n_res_channel, stride=2
        )
        self.quantize_conv_b = nn.Conv2d(embed_dim + channel, embed_dim, 1)
        self.quantize_b = Quantize(embed_dim, n_embed)
        self.upsample_t = nn.ConvTranspose2d(
            embed_dim, embed_dim, 4, stride=2, padding=1
        )
        self.dec = Decoder(
            embed_dim + embed_dim,
            in_channel,
            channel,
            n_res_block,
            n_res_channel,
            stride=4,
        )

        kernels = [64, 128, 256, 512]

        self.src_ch = src_ch
        self.inplanes = kernels[0]
        self.stem_conv = nn.Conv2d(3, kernels[0], kernel_size=7,
                                   stride=2, padding=3, bias=False)

        self.stem_conv_edge = nn.Conv2d(1, kernels[0], kernel_size=7,
                                        stride=2, padding=3, bias=False)
        self.stem_norm_edge = nn.InstanceNorm2d(kernels[0])
        self.relu_edge = nn.ReLU(inplace=True)

        print("####1 BNmode => ", BNmode)
        if BNmode == 'BN':
            self.stem_norm = nn.BatchNorm2d(kernels[0])
        elif BNmode == 'IN':
            self.stem_norm = nn.InstanceNorm2d(kernels[0])
        elif BNmode == 'GN':
            self.stem_norm = nn.GroupNorm(32, kernels[0])
        # Adding AdaIN, combined BN and IN
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(block, kernels[0], layers[0], stride=1, BNmode=BNmode)
        self.layer2 = self._make_layer(block, kernels[1], layers[1], stride=2, BNmode=BNmode)
        self.layer3 = self._make_layer(block, kernels[2], layers[2], stride=2, BNmode=BNmode)
        self.inplanes = kernels[2] + kernels[2]
        self.layer4 = self._make_layer(block, kernels[3], layers[3], stride=2, BNmode=BNmode)

        # L components
        self.deconvL1 = upconv(kernels[3], kernels[2], ratio="x2")
        self.inplanes = kernels[2]
        self.upResL1 = self._make_layer(block, kernels[2], layers[2], BNmode=BNmode)
        self.deconvL2 = upconv(kernels[2], kernels[1], ratio="x2")
        self.inplanes = kernels[1]
        self.upResL2 = self._make_layer(block, kernels[1], layers[1], BNmode=BNmode)
        self.deconvL3 = upconv(kernels[1], kernels[0], ratio="x2")
        # edge parts
        self.inplanes = kernels[0] + kernels[0]
        self.upResL3 = self._make_layer(block, kernels[0], layers[0], BNmode=BNmode)
        self.deconvL4 = upconv(kernels[0], kernels[0], ratio="x2")
        self.predL = nn.Conv2d(kernels[0], tar_ch[0], kernel_size=3, stride=1, padding=1, bias=False)

        # a components
        self.deconva1 = upconv(kernels[3], kernels[2], ratio="x2")
        self.inplanes = kernels[2]
        self.upResa1 = self._make_layer(block, kernels[2], layers[2], BNmode=BNmode)
        self.deconva2 = upconv(kernels[2], kernels[1], ratio="x2")
        self.inplanes = kernels[1]
        self.upResa2 = self._make_layer(block, kernels[1], layers[1], BNmode=BNmode)
        self.deconva3 = upconv(kernels[1], kernels[0], ratio="x2")
        self.inplanes = kernels[0]
        self.upResa3 = self._make_layer(block, kernels[0], layers[0], BNmode=BNmode)
        self.deconva4 = upconv(kernels[0], kernels[0], ratio="x2")
        self.preda = nn.Conv2d(kernels[0], tar_ch[1], kernel_size=3, stride=1, padding=1, bias=False)

        # b components
        self.deconvb1 = upconv(kernels[3], kernels[2], ratio="x2")
        self.inplanes = kernels[2]
        self.upResb1 = self._make_layer(block, kernels[2], layers[2], BNmode=BNmode)
        self.deconvb2 = upconv(kernels[2], kernels[1], ratio="x2")
        self.inplanes = kernels[1]
        self.upResb2 = self._make_layer(block, kernels[1], layers[1], BNmode=BNmode)
        self.deconvb3 = upconv(kernels[1], kernels[0], ratio="x2")
        self.inplanes = kernels[0]
        self.upResb3 = self._make_layer(block, kernels[0], layers[0], BNmode=BNmode)
        self.deconvb4 = upconv(kernels[0], kernels[0], ratio="x2")
        self.predb = nn.Conv2d(kernels[0], tar_ch[2], kernel_size=3, stride=1, padding=1, bias=False)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def encode(self, input):
        enc_b = self.enc_b(input)
        enc_t = self.enc_t(enc_b)
        quant_t = self.quantize_conv_t(enc_t).permute(0, 2, 3, 1)
        quant_t, diff_t, id_t = self.quantize_t(quant_t)
        quant_t = quant_t.permute(0, 3, 1, 2)
        diff_t = diff_t.unsqueeze(0)

        dec_t = self.dec_t(quant_t)
        enc_b = torch.cat([dec_t, enc_b], 1)

        quant_b = self.quantize_conv_b(enc_b).permute(0, 2, 3, 1)
        quant_b, diff_b, id_b = self.quantize_b(quant_b)
        quant_b = quant_b.permute(0, 3, 1, 2)
        diff_b = diff_b.unsqueeze(0)
        # enc_b ?
        return enc_t, quant_t, quant_b, diff_t + diff_b, id_t, id_b

    def decode(self, quant_t, quant_b):
        upsample_t = self.upsample_t(quant_t)
        quant = torch.cat([upsample_t, quant_b], 1)
        dec = self.dec(quant)

        return dec

    def _make_layer(self, block, planes, blocks, stride=1, BNmode='BN'):
        downsample = None
        print("####2 BNmode => ", BNmode)
        if stride != 1 or self.inplanes != planes * block.expansion:
            bnl = nn.BatchNorm2d(planes * block.expansion)
            if BNmode == 'IN':
                bnl = nn.InstanceNorm2d(planes * block.expansion)
            if BNmode == 'GN':
                bnl = nn.GroupNorm(32, planes * block.expansion)
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                bnl,
            )
            # Adding AdaIN

        layers = []
        if BNmode == 'GN':
            layers.append(block(self.inplanes, planes, stride, downsample, BNmode=BNmode))
        else:
            layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            if BNmode == 'GN':
                layers.append(block(self.inplanes, planes, BNmode=BNmode))
            else:
                layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    @staticmethod
    def _gaussian_filter(channel, kernel_size, sigma=1.5):
        gauss = torch.Tensor(
            [math.exp(-(x - kernel_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(kernel_size)])
        gauss_1D = (gauss / gauss.sum()).unsqueeze(1)
        gauss_2D = gauss_1D.mm(gauss_1D.t()).float().unsqueeze(0).unsqueeze(0)
        return gauss_2D.expand(channel, 1, kernel_size, kernel_size).contiguous()

    def my_cosine_sync(self, x1, x2, x3, kernel_size=3, sigma=1.5):
        nb, ch, row, col = x1.size()
        cosine_1 = F.cosine_similarity(x1, x2, dim=1).unsqueeze(1)
        cosine_2 = F.cosine_similarity(x1, x3, dim=1).unsqueeze(1)
        cosine = cosine_1 + cosine_2
        filters = self._gaussian_filter(1, kernel_size, sigma).to(cosine.device)
        attenMap = F.conv2d(cosine, filters, padding=1)
        attenMap = attenMap.expand(nb, ch, row, col).contiguous()
        x2 = x2 * attenMap
        x3 = x3 * attenMap
        return x1, x2, x3

    def _status_sync(self, x1, x2, kernel_size=3, sigma=1.5):
        nb, ch, row, col = x1.size()
        cosine = F.cosine_similarity(x1, x2, dim=1).unsqueeze(1)
        filters = self._gaussian_filter(1, kernel_size, sigma).to(cosine.device)
        # print("Cosine =>", cosine.size(), "Filter =>", filters.size())
        attenMap = F.conv2d(cosine, filters, padding=1)
        attenMap = attenMap.expand(nb, ch, row, col).contiguous()
        x1 = x1 * attenMap
        x2 = x2 * attenMap
        return x1, x2

    def forward(self, x):
        if self.src_ch == 1:
            x = torch.cat([x, x, x], dim=1)
        # forward

        _, _, _, _, _, thin_edges = self.edge_detection(x)

        enc_t, quant_t, quant_b, diff, _, _ = self.encode(x)
        dec = self.decode(quant_t, quant_b)

        # edge parts
        canny_edge = self.stem_conv_edge(thin_edges)
        canny_edge = self.stem_norm_edge(canny_edge)
        canny_edge = self.relu_edge(canny_edge)

        x1 = self.stem_conv(x)
        x1 = self.stem_norm(x1)
        x2 = self.relu(x1)

        # Encoder
        x3 = self.layer1(x2)
        x4 = self.layer2(x3)
        x5 = self.layer3(x4)
        x5_cat = torch.cat([x5, quant_t], dim=1)
        x6 = self.layer4(x5_cat)

        ## stage 1
        x7_L = self.deconvL1(x6)
        x7_a = self.deconva1(x6)
        x7_b = self.deconvb1(x6)
        # sync
        x7_L, x7_a, x7_b = self.my_cosine_sync(x7_L, x7_a, x7_b)

        x7_L = x7_L + x5
        x7_a = x7_a + x5
        x7_b = x7_b + x5

        ## stage 2
        x8_L = self.upResL1(x7_L)
        x8_L = self.deconvL2(x8_L)
        x8_a = self.upResa1(x7_a)
        x8_a = self.deconva2(x8_a)
        x8_b = self.upResb1(x7_b)
        x8_b = self.deconvb2(x8_b)
        # sync
        x8_L, x8_a, x8_b = self.my_cosine_sync(x8_L, x8_a, x8_b)

        x8_L = x8_L + x4
        x8_a = x8_a + x4
        x8_b = x8_b + x4

        ## stage 3
        x9_L = self.upResL2(x8_L)
        x9_L = self.deconvL3(x9_L)
        x9_a = self.upResa2(x8_a)
        x9_a = self.deconva3(x9_a)
        x9_b = self.upResb2(x8_b)
        x9_b = self.deconvb3(x9_b)

        # sync
        x9_L, x9_a, x9_b = self.my_cosine_sync(x9_L, x9_a, x9_b)

        x9_L = x9_L + x3
        x9_a = x9_a + x3
        x9_b = x9_b + x3
        # edge
        x9_L = torch.cat([x9_L, canny_edge], dim=1)
        ## stage 4
        x10_L = self.upResL3(x9_L)
        x10_L = self.deconvL4(x10_L)
        x10_a = self.upResa3(x9_a)
        x10_a = self.deconva4(x10_a)
        x10_b = self.upResb3(x9_b)
        x10_b = self.deconvb4(x10_b)
        # sync
        x10_L, x10_a, x10_b = self.my_cosine_sync(x10_L, x10_a, x10_b)

        ## Final stage
        x_final_L = self.predL(x10_L)
        x_final_a = self.preda(x10_a)
        x_final_b = self.predb(x10_b)

        return x_final_L, x_final_a, x_final_b, dec, diff


class ResNet_pix2pixHD_Cannyedge_Custom(nn.Module):

    def __init__(self, block, layers, src_ch, tar_ch, BNmode,
                 in_channel=3, channel=128, n_res_block=2, n_res_channel=32, embed_dim=256, n_embed=512):
        super(ResNet_pix2pixHD_Cannyedge_Custom, self).__init__()
        self.enc_b = Encoder(in_channel, channel, n_res_block, n_res_channel, stride=4)
        self.enc_t = Encoder(channel, channel, n_res_block, n_res_channel, stride=2)
        self.quantize_conv_t = nn.Conv2d(channel, embed_dim, 1)
        self.quantize_t = Quantize(embed_dim, n_embed)
        self.dec_t = Decoder(
            embed_dim, embed_dim, channel, n_res_block, n_res_channel, stride=2
        )
        self.quantize_conv_b = nn.Conv2d(embed_dim + channel, embed_dim, 1)
        self.quantize_b = Quantize(embed_dim, n_embed)
        self.upsample_t = nn.ConvTranspose2d(
            embed_dim, embed_dim, 4, stride=2, padding=1
        )
        self.dec = Decoder(
            embed_dim + embed_dim,
            in_channel,
            channel,
            n_res_block,
            n_res_channel,
            stride=4,
        )

        kernels = [64, 128, 256, 512]

        self.src_ch = src_ch
        self.inplanes = kernels[0]
        self.stem_conv = nn.Conv2d(3, kernels[0], kernel_size=7,
                                   stride=2, padding=3, bias=False)

        print("####1 BNmode => ", BNmode)
        if BNmode == 'BN':
            self.stem_norm = nn.BatchNorm2d(kernels[0])
        elif BNmode == 'IN':
            self.stem_norm = nn.InstanceNorm2d(kernels[0])
        elif BNmode == 'GN':
            self.stem_norm = nn.GroupNorm(32, kernels[0])
        # Adding AdaIN, combined BN and IN
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(block, kernels[0], layers[0], stride=1, BNmode=BNmode)
        self.layer2 = self._make_layer(block, kernels[1], layers[1], stride=2, BNmode=BNmode)
        self.layer3 = self._make_layer(block, kernels[2], layers[2], stride=2, BNmode=BNmode)
        self.inplanes = kernels[2] + kernels[2]
        self.layer4 = self._make_layer(block, kernels[3], layers[3], stride=2, BNmode=BNmode)

        # L components
        self.deconvL1 = upconv(kernels[3], kernels[2], ratio="x2")
        self.inplanes = kernels[2]
        self.upResL1 = self._make_layer(block, kernels[2], layers[2], BNmode=BNmode)
        self.deconvL2 = upconv(kernels[2], kernels[1], ratio="x2")
        self.inplanes = kernels[1]
        self.upResL2 = self._make_layer(block, kernels[1], layers[1], BNmode=BNmode)
        self.deconvL3 = upconv(kernels[1], kernels[0], ratio="x2")
        self.inplanes = kernels[0]
        self.upResL3 = self._make_layer(block, kernels[0], layers[0], BNmode=BNmode)
        self.deconvL4 = upconv(kernels[0], kernels[0], ratio="x2")
        self.predL = nn.Conv2d(kernels[0], tar_ch[0], kernel_size=3, stride=1, padding=1, bias=False)

        # a components
        self.deconva1 = upconv(kernels[3], kernels[2], ratio="x2")
        self.inplanes = kernels[2]
        self.upResa1 = self._make_layer(block, kernels[2], layers[2], BNmode=BNmode)
        self.deconva2 = upconv(kernels[2], kernels[1], ratio="x2")
        self.inplanes = kernels[1]
        self.upResa2 = self._make_layer(block, kernels[1], layers[1], BNmode=BNmode)
        self.deconva3 = upconv(kernels[1], kernels[0], ratio="x2")
        self.inplanes = kernels[0]
        self.upResa3 = self._make_layer(block, kernels[0], layers[0], BNmode=BNmode)
        self.deconva4 = upconv(kernels[0], kernels[0], ratio="x2")
        self.preda = nn.Conv2d(kernels[0], tar_ch[1], kernel_size=3, stride=1, padding=1, bias=False)

        # b components
        self.deconvb1 = upconv(kernels[3], kernels[2], ratio="x2")
        self.inplanes = kernels[2]
        self.upResb1 = self._make_layer(block, kernels[2], layers[2], BNmode=BNmode)
        self.deconvb2 = upconv(kernels[2], kernels[1], ratio="x2")
        self.inplanes = kernels[1]
        self.upResb2 = self._make_layer(block, kernels[1], layers[1], BNmode=BNmode)
        self.deconvb3 = upconv(kernels[1], kernels[0], ratio="x2")
        self.inplanes = kernels[0]
        self.upResb3 = self._make_layer(block, kernels[0], layers[0], BNmode=BNmode)
        self.deconvb4 = upconv(kernels[0], kernels[0], ratio="x2")
        self.predb = nn.Conv2d(kernels[0], tar_ch[2], kernel_size=3, stride=1, padding=1, bias=False)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def encode(self, input):
        enc_b = self.enc_b(input)
        enc_t = self.enc_t(enc_b)
        quant_t = self.quantize_conv_t(enc_t).permute(0, 2, 3, 1)
        quant_t, diff_t, id_t = self.quantize_t(quant_t)
        quant_t = quant_t.permute(0, 3, 1, 2)
        diff_t = diff_t.unsqueeze(0)

        dec_t = self.dec_t(quant_t)
        enc_b = torch.cat([dec_t, enc_b], 1)

        quant_b = self.quantize_conv_b(enc_b).permute(0, 2, 3, 1)
        quant_b, diff_b, id_b = self.quantize_b(quant_b)
        quant_b = quant_b.permute(0, 3, 1, 2)
        diff_b = diff_b.unsqueeze(0)
        # enc_b ?
        return enc_t, quant_t, quant_b, diff_t + diff_b, id_t, id_b

    def decode(self, quant_t, quant_b):
        upsample_t = self.upsample_t(quant_t)
        quant = torch.cat([upsample_t, quant_b], 1)
        dec = self.dec(quant)

        return dec

    def _make_layer(self, block, planes, blocks, stride=1, BNmode='BN'):
        downsample = None
        print("####2 BNmode => ", BNmode)
        if stride != 1 or self.inplanes != planes * block.expansion:
            bnl = nn.BatchNorm2d(planes * block.expansion)
            if BNmode == 'IN':
                bnl = nn.InstanceNorm2d(planes * block.expansion)
            if BNmode == 'GN':
                bnl = nn.GroupNorm(32, planes * block.expansion)
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                bnl,
            )
            # Adding AdaIN

        layers = []
        if BNmode == 'GN':
            layers.append(block(self.inplanes, planes, stride, downsample, BNmode=BNmode))
        else:
            layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            if BNmode == 'GN':
                layers.append(block(self.inplanes, planes, BNmode=BNmode))
            else:
                layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    @staticmethod
    def _gaussian_filter(channel, kernel_size, sigma=1.5):
        gauss = torch.Tensor(
            [math.exp(-(x - kernel_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(kernel_size)])
        gauss_1D = (gauss / gauss.sum()).unsqueeze(1)
        gauss_2D = gauss_1D.mm(gauss_1D.t()).float().unsqueeze(0).unsqueeze(0)
        return gauss_2D.expand(channel, 1, kernel_size, kernel_size).contiguous()

    def my_cosine_sync(self, x1, x2, x3, kernel_size=3, sigma=1.5):
        nb, ch, row, col = x1.size()
        cosine_1 = F.cosine_similarity(x1, x2, dim=1).unsqueeze(1)
        cosine_2 = F.cosine_similarity(x1, x3, dim=1).unsqueeze(1)
        cosine = cosine_1 + cosine_2
        filters = self._gaussian_filter(1, kernel_size, sigma).to(cosine.device)
        attenMap = F.conv2d(cosine, filters, padding=1)
        attenMap = attenMap.expand(nb, ch, row, col).contiguous()
        x2 = x2 * attenMap
        x3 = x3 * attenMap
        return x1, x2, x3

    def _status_sync(self, x1, x2, kernel_size=3, sigma=1.5):
        nb, ch, row, col = x1.size()
        cosine = F.cosine_similarity(x1, x2, dim=1).unsqueeze(1)
        filters = self._gaussian_filter(1, kernel_size, sigma).to(cosine.device)
        # print("Cosine =>", cosine.size(), "Filter =>", filters.size())
        attenMap = F.conv2d(cosine, filters, padding=1)
        attenMap = attenMap.expand(nb, ch, row, col).contiguous()
        x1 = x1 * attenMap
        x2 = x2 * attenMap
        return x1, x2

    def forward(self, x):
        if self.src_ch == 1:
            x = torch.cat([x, x, x], dim=1)
        # forward

        enc_t, quant_t, quant_b, diff, _, _ = self.encode(x)
        dec = self.decode(quant_t, quant_b)

        # edge parts

        x1 = self.stem_conv(x)
        x1 = self.stem_norm(x1)
        x2 = self.relu(x1)

        # Encoder
        x3 = self.layer1(x2)
        x4 = self.layer2(x3)
        x5 = self.layer3(x4)
        x5_cat = torch.cat([x5, quant_t], dim=1)
        x6 = self.layer4(x5_cat)

        ## stage 1
        x7_L = self.deconvL1(x6)
        x7_a = self.deconva1(x6)
        x7_b = self.deconvb1(x6)
        # sync
        x7_L, x7_a, x7_b = self.my_cosine_sync(x7_L, x7_a, x7_b)

        x7_L = x7_L + x5
        x7_a = x7_a + x5
        x7_b = x7_b + x5

        ## stage 2
        x8_L = self.upResL1(x7_L)
        x8_L = self.deconvL2(x8_L)
        x8_a = self.upResa1(x7_a)
        x8_a = self.deconva2(x8_a)
        x8_b = self.upResb1(x7_b)
        x8_b = self.deconvb2(x8_b)
        # sync
        x8_L, x8_a, x8_b = self.my_cosine_sync(x8_L, x8_a, x8_b)

        x8_L = x8_L + x4
        x8_a = x8_a + x4
        x8_b = x8_b + x4

        ## stage 3
        x9_L = self.upResL2(x8_L)
        x9_L = self.deconvL3(x9_L)
        x9_a = self.upResa2(x8_a)
        x9_a = self.deconva3(x9_a)
        x9_b = self.upResb2(x8_b)
        x9_b = self.deconvb3(x9_b)

        # sync
        x9_L, x9_a, x9_b = self.my_cosine_sync(x9_L, x9_a, x9_b)

        x9_L = x9_L + x3
        x9_a = x9_a + x3
        x9_b = x9_b + x3

        ## stage 4
        x10_L = self.upResL3(x9_L)
        x10_L = self.deconvL4(x10_L)
        x10_a = self.upResa3(x9_a)
        x10_a = self.deconva4(x10_a)
        x10_b = self.upResb3(x9_b)
        x10_b = self.deconvb4(x10_b)
        # sync
        x10_L, x10_a, x10_b = self.my_cosine_sync(x10_L, x10_a, x10_b)

        ## Final stage
        x_final_L = self.predL(x10_L)
        x_final_a = self.preda(x10_a)
        x_final_b = self.predb(x10_b)

        return x_final_L, x_final_a, x_final_b, dec, diff


class ResNet_pix2pixHD_Cannyedge_Custom_edge(nn.Module):

    def __init__(self, block, layers, src_ch, tar_ch, BNmode,
                 in_channel=3, channel=128, n_res_block=2, n_res_channel=32, embed_dim=256, n_embed=512):
        super(ResNet_pix2pixHD_Cannyedge_Custom_edge, self).__init__()

        self.edge_detection = CannyFilter(use_cuda=True)
        self.enc_b = Encoder(in_channel, channel, n_res_block, n_res_channel, stride=4)
        self.enc_t = Encoder(channel, channel, n_res_block, n_res_channel, stride=2)
        self.quantize_conv_t = nn.Conv2d(channel, embed_dim, 1)
        self.quantize_t = Quantize(embed_dim, n_embed)
        self.dec_t = Decoder(
            embed_dim, embed_dim, channel, n_res_block, n_res_channel, stride=2
        )
        self.quantize_conv_b = nn.Conv2d(embed_dim + channel, embed_dim, 1)
        self.quantize_b = Quantize(embed_dim, n_embed)
        self.upsample_t = nn.ConvTranspose2d(
            embed_dim, embed_dim, 4, stride=2, padding=1
        )
        self.dec = Decoder(
            embed_dim + embed_dim,
            in_channel,
            channel,
            n_res_block,
            n_res_channel,
            stride=4,
        )

        kernels = [64, 128, 256, 512]

        self.src_ch = src_ch
        self.inplanes = kernels[0]

        self.stem_conv_edge = nn.Conv2d(3, kernels[0], kernel_size=7,
                                        stride=2, padding=3, bias=False)
        self.stem_norm_edge = nn.InstanceNorm2d(kernels[0])
        self.relu_edge = nn.ReLU(inplace=True)

        print("####1 BNmode => ", BNmode)
        if BNmode == 'BN':
            self.stem_norm = nn.BatchNorm2d(kernels[0])
        elif BNmode == 'IN':
            self.stem_norm = nn.InstanceNorm2d(kernels[0])
        elif BNmode == 'GN':
            self.stem_norm = nn.GroupNorm(32, kernels[0])
        # Adding AdaIN, combined BN and IN
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(block, kernels[0], layers[0], stride=1, BNmode=BNmode)
        self.layer2 = self._make_layer(block, kernels[1], layers[1], stride=2, BNmode=BNmode)
        self.layer3 = self._make_layer(block, kernels[2], layers[2], stride=2, BNmode=BNmode)
        self.inplanes = kernels[2] + kernels[2]
        self.layer4 = self._make_layer(block, kernels[3], layers[3], stride=2, BNmode=BNmode)

        # L components
        self.deconvL1 = upconv(kernels[3], kernels[2], ratio="x2")
        self.inplanes = kernels[2]
        self.upResL1 = self._make_layer(block, kernels[2], layers[2], BNmode=BNmode)
        self.deconvL2 = upconv(kernels[2], kernels[1], ratio="x2")
        self.inplanes = kernels[1]
        self.upResL2 = self._make_layer(block, kernels[1], layers[1], BNmode=BNmode)
        self.deconvL3 = upconv(kernels[1], kernels[0], ratio="x2")
        self.inplanes = kernels[0]
        self.upResL3 = self._make_layer(block, kernels[0], layers[0], BNmode=BNmode)
        self.deconvL4 = upconv(kernels[0], kernels[0], ratio="x2")
        self.predL = nn.Conv2d(kernels[0], tar_ch[0], kernel_size=3, stride=1, padding=1, bias=False)

        # a components
        self.deconva1 = upconv(kernels[3], kernels[2], ratio="x2")
        self.inplanes = kernels[2]
        self.upResa1 = self._make_layer(block, kernels[2], layers[2], BNmode=BNmode)
        self.deconva2 = upconv(kernels[2], kernels[1], ratio="x2")
        self.inplanes = kernels[1]
        self.upResa2 = self._make_layer(block, kernels[1], layers[1], BNmode=BNmode)
        self.deconva3 = upconv(kernels[1], kernels[0], ratio="x2")
        self.inplanes = kernels[0]
        self.upResa3 = self._make_layer(block, kernels[0], layers[0], BNmode=BNmode)
        self.deconva4 = upconv(kernels[0], kernels[0], ratio="x2")
        self.preda = nn.Conv2d(kernels[0], tar_ch[1], kernel_size=3, stride=1, padding=1, bias=False)

        # b components
        self.deconvb1 = upconv(kernels[3], kernels[2], ratio="x2")
        self.inplanes = kernels[2]
        self.upResb1 = self._make_layer(block, kernels[2], layers[2], BNmode=BNmode)
        self.deconvb2 = upconv(kernels[2], kernels[1], ratio="x2")
        self.inplanes = kernels[1]
        self.upResb2 = self._make_layer(block, kernels[1], layers[1], BNmode=BNmode)
        self.deconvb3 = upconv(kernels[1], kernels[0], ratio="x2")
        self.inplanes = kernels[0]
        self.upResb3 = self._make_layer(block, kernels[0], layers[0], BNmode=BNmode)
        self.deconvb4 = upconv(kernels[0], kernels[0], ratio="x2")
        self.predb = nn.Conv2d(kernels[0], tar_ch[2], kernel_size=3, stride=1, padding=1, bias=False)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def encode(self, input):
        enc_b = self.enc_b(input)
        enc_t = self.enc_t(enc_b)
        quant_t = self.quantize_conv_t(enc_t).permute(0, 2, 3, 1)
        quant_t, diff_t, id_t = self.quantize_t(quant_t)
        quant_t = quant_t.permute(0, 3, 1, 2)
        diff_t = diff_t.unsqueeze(0)

        dec_t = self.dec_t(quant_t)
        enc_b = torch.cat([dec_t, enc_b], 1)

        quant_b = self.quantize_conv_b(enc_b).permute(0, 2, 3, 1)
        quant_b, diff_b, id_b = self.quantize_b(quant_b)
        quant_b = quant_b.permute(0, 3, 1, 2)
        diff_b = diff_b.unsqueeze(0)
        # enc_b ?
        return enc_t, quant_t, quant_b, diff_t + diff_b, id_t, id_b

    def decode(self, quant_t, quant_b):
        upsample_t = self.upsample_t(quant_t)
        quant = torch.cat([upsample_t, quant_b], 1)
        dec = self.dec(quant)

        return dec

    def _make_layer(self, block, planes, blocks, stride=1, BNmode='BN'):
        downsample = None
        print("####2 BNmode => ", BNmode)
        if stride != 1 or self.inplanes != planes * block.expansion:
            bnl = nn.BatchNorm2d(planes * block.expansion)
            if BNmode == 'IN':
                bnl = nn.InstanceNorm2d(planes * block.expansion)
            if BNmode == 'GN':
                bnl = nn.GroupNorm(32, planes * block.expansion)
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                bnl,
            )
            # Adding AdaIN

        layers = []
        if BNmode == 'GN':
            layers.append(block(self.inplanes, planes, stride, downsample, BNmode=BNmode))
        else:
            layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            if BNmode == 'GN':
                layers.append(block(self.inplanes, planes, BNmode=BNmode))
            else:
                layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    @staticmethod
    def _gaussian_filter(channel, kernel_size, sigma=1.5):
        gauss = torch.Tensor(
            [math.exp(-(x - kernel_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(kernel_size)])
        gauss_1D = (gauss / gauss.sum()).unsqueeze(1)
        gauss_2D = gauss_1D.mm(gauss_1D.t()).float().unsqueeze(0).unsqueeze(0)
        return gauss_2D.expand(channel, 1, kernel_size, kernel_size).contiguous()

    def my_cosine_sync(self, x1, x2, x3, kernel_size=3, sigma=1.5):
        nb, ch, row, col = x1.size()
        cosine_1 = F.cosine_similarity(x1, x2, dim=1).unsqueeze(1)
        cosine_2 = F.cosine_similarity(x1, x3, dim=1).unsqueeze(1)
        cosine = cosine_1 + cosine_2
        filters = self._gaussian_filter(1, kernel_size, sigma).to(cosine.device)
        attenMap = F.conv2d(cosine, filters, padding=1)
        attenMap = attenMap.expand(nb, ch, row, col).contiguous()
        x2 = x2 * attenMap
        x3 = x3 * attenMap
        return x1, x2, x3

    def _status_sync(self, x1, x2, kernel_size=3, sigma=1.5):
        nb, ch, row, col = x1.size()
        cosine = F.cosine_similarity(x1, x2, dim=1).unsqueeze(1)
        filters = self._gaussian_filter(1, kernel_size, sigma).to(cosine.device)
        # print("Cosine =>", cosine.size(), "Filter =>", filters.size())
        attenMap = F.conv2d(cosine, filters, padding=1)
        attenMap = attenMap.expand(nb, ch, row, col).contiguous()
        x1 = x1 * attenMap
        x2 = x2 * attenMap
        return x1, x2

    def forward(self, x):
        if self.src_ch == 1:
            x = torch.cat([x, x, x], dim=1)
        # forward

        _, _, _, _, _, thin_edges = self.edge_detection(x)
        thin_edges = torch.cat([thin_edges, thin_edges, thin_edges], dim=1)
        enc_t, quant_t, quant_b, diff, _, _ = self.encode(x)
        dec = self.decode(quant_t, quant_b)

        # edge parts
        canny_edge = self.stem_conv_edge(thin_edges)
        canny_edge = self.stem_norm_edge(canny_edge)
        canny_edge = self.relu_edge(canny_edge)

        # Encoder
        x3 = self.layer1(canny_edge)
        x4 = self.layer2(x3)
        x5 = self.layer3(x4)
        x5_cat = torch.cat([x5, quant_t], dim=1)
        x6 = self.layer4(x5_cat)

        ## stage 1
        x7_L = self.deconvL1(x6)
        x7_a = self.deconva1(x6)
        x7_b = self.deconvb1(x6)
        # sync
        x7_L, x7_a, x7_b = self.my_cosine_sync(x7_L, x7_a, x7_b)

        x7_L = x7_L + x5
        x7_a = x7_a + x5
        x7_b = x7_b + x5

        ## stage 2
        x8_L = self.upResL1(x7_L)
        x8_L = self.deconvL2(x8_L)
        x8_a = self.upResa1(x7_a)
        x8_a = self.deconva2(x8_a)
        x8_b = self.upResb1(x7_b)
        x8_b = self.deconvb2(x8_b)
        # sync
        x8_L, x8_a, x8_b = self.my_cosine_sync(x8_L, x8_a, x8_b)

        x8_L = x8_L + x4
        x8_a = x8_a + x4
        x8_b = x8_b + x4

        ## stage 3
        x9_L = self.upResL2(x8_L)
        x9_L = self.deconvL3(x9_L)
        x9_a = self.upResa2(x8_a)
        x9_a = self.deconva3(x9_a)
        x9_b = self.upResb2(x8_b)
        x9_b = self.deconvb3(x9_b)

        # sync
        x9_L, x9_a, x9_b = self.my_cosine_sync(x9_L, x9_a, x9_b)

        x9_L = x9_L + x3
        x9_a = x9_a + x3
        x9_b = x9_b + x3

        ## stage 4
        x10_L = self.upResL3(x9_L)
        x10_L = self.deconvL4(x10_L)
        x10_a = self.upResa3(x9_a)
        x10_a = self.deconva4(x10_a)
        x10_b = self.upResb3(x9_b)
        x10_b = self.deconvb4(x10_b)
        # sync
        x10_L, x10_a, x10_b = self.my_cosine_sync(x10_L, x10_a, x10_b)

        ## Final stage
        x_final_L = self.predL(x10_L)
        x_final_a = self.preda(x10_a)
        x_final_b = self.predb(x10_b)

        return x_final_L, x_final_a, x_final_b, dec, diff


class ResNet_pix2pixHD_Cannyedge_Decoder(nn.Module):

    def __init__(self, block, layers, src_ch, tar_ch, BNmode,
                 in_channel=3, channel=128, n_res_block=2, n_res_channel=32, embed_dim=256, n_embed=512):
        super(ResNet_pix2pixHD_Cannyedge_Decoder, self).__init__()

        self.enc_b = Encoder(6, channel, n_res_block, n_res_channel, stride=4)
        self.enc_t = Encoder(channel, channel, n_res_block, n_res_channel, stride=2)
        self.quantize_conv_t = nn.Conv2d(channel, embed_dim, 1)
        self.quantize_t = Quantize(embed_dim, n_embed)
        self.dec_t = Decoder(
            embed_dim, embed_dim, channel, n_res_block, n_res_channel, stride=2
        )
        self.quantize_conv_b = nn.Conv2d(embed_dim + channel, embed_dim, 1)
        self.quantize_b = Quantize(embed_dim, n_embed)
        self.upsample_t = nn.ConvTranspose2d(
            embed_dim, embed_dim, 4, stride=2, padding=1
        )
        self.dec = Decoder(
            embed_dim + embed_dim,
            in_channel,
            channel,
            n_res_block,
            n_res_channel,
            stride=4,
        )

        kernels = [64, 128, 256, 512]

        self.src_ch = src_ch
        self.inplanes = kernels[0]
        self.stem_conv = nn.Conv2d(6, kernels[0], kernel_size=7,
                                   stride=2, padding=3, bias=False)

        print("####1 BNmode => ", BNmode)
        if BNmode == 'BN':
            self.stem_norm = nn.BatchNorm2d(kernels[0])
        elif BNmode == 'IN':
            self.stem_norm = nn.InstanceNorm2d(kernels[0])
        elif BNmode == 'GN':
            self.stem_norm = nn.GroupNorm(32, kernels[0])
        # Adding AdaIN, combined BN and IN
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(block, kernels[0], layers[0], stride=1, BNmode=BNmode)
        self.layer2 = self._make_layer(block, kernels[1], layers[1], stride=2, BNmode=BNmode)
        self.layer3 = self._make_layer(block, kernels[2], layers[2], stride=2, BNmode=BNmode)
        self.inplanes = kernels[2] + kernels[2]
        self.layer4 = self._make_layer(block, kernels[3], layers[3], stride=2, BNmode=BNmode)

        # L components
        self.deconvL1 = upconv(kernels[3], kernels[2], ratio="x2")
        self.inplanes = kernels[2]
        self.upResL1 = self._make_layer(block, kernels[2], layers[2], BNmode=BNmode)
        self.deconvL2 = upconv(kernels[2], kernels[1], ratio="x2")
        self.inplanes = kernels[1]
        self.upResL2 = self._make_layer(block, kernels[1], layers[1], BNmode=BNmode)
        self.deconvL3 = upconv(kernels[1], kernels[0], ratio="x2")
        self.inplanes = kernels[0]
        self.upResL3 = self._make_layer(block, kernels[0], layers[0], BNmode=BNmode)
        self.deconvL4 = upconv(kernels[0], kernels[0], ratio="x2")
        self.predL = nn.Conv2d(kernels[0], tar_ch[0], kernel_size=3, stride=1, padding=1, bias=False)

        # a components
        self.deconva1 = upconv(kernels[3], kernels[2], ratio="x2")
        self.inplanes = kernels[2]
        self.upResa1 = self._make_layer(block, kernels[2], layers[2], BNmode=BNmode)
        self.deconva2 = upconv(kernels[2], kernels[1], ratio="x2")
        self.inplanes = kernels[1]
        self.upResa2 = self._make_layer(block, kernels[1], layers[1], BNmode=BNmode)
        self.deconva3 = upconv(kernels[1], kernels[0], ratio="x2")
        self.inplanes = kernels[0]
        self.upResa3 = self._make_layer(block, kernels[0], layers[0], BNmode=BNmode)
        self.deconva4 = upconv(kernels[0], kernels[0], ratio="x2")
        self.preda = nn.Conv2d(kernels[0], tar_ch[1], kernel_size=3, stride=1, padding=1, bias=False)

        # b components
        self.deconvb1 = upconv(kernels[3], kernels[2], ratio="x2")
        self.inplanes = kernels[2]
        self.upResb1 = self._make_layer(block, kernels[2], layers[2], BNmode=BNmode)
        self.deconvb2 = upconv(kernels[2], kernels[1], ratio="x2")
        self.inplanes = kernels[1]
        self.upResb2 = self._make_layer(block, kernels[1], layers[1], BNmode=BNmode)
        self.deconvb3 = upconv(kernels[1], kernels[0], ratio="x2")
        self.inplanes = kernels[0]
        self.upResb3 = self._make_layer(block, kernels[0], layers[0], BNmode=BNmode)
        self.deconvb4 = upconv(kernels[0], kernels[0], ratio="x2")
        self.predb = nn.Conv2d(kernels[0], tar_ch[2], kernel_size=3, stride=1, padding=1, bias=False)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def encode(self, input):
        enc_b = self.enc_b(input)
        enc_t = self.enc_t(enc_b)
        quant_t = self.quantize_conv_t(enc_t).permute(0, 2, 3, 1)
        quant_t, diff_t, id_t = self.quantize_t(quant_t)
        quant_t = quant_t.permute(0, 3, 1, 2)
        diff_t = diff_t.unsqueeze(0)

        dec_t = self.dec_t(quant_t)
        enc_b = torch.cat([dec_t, enc_b], 1)

        quant_b = self.quantize_conv_b(enc_b).permute(0, 2, 3, 1)
        quant_b, diff_b, id_b = self.quantize_b(quant_b)
        quant_b = quant_b.permute(0, 3, 1, 2)
        diff_b = diff_b.unsqueeze(0)
        # enc_b ?
        return enc_t, quant_t, quant_b, diff_t + diff_b, id_t, id_b

    def decode(self, quant_t, quant_b):
        upsample_t = self.upsample_t(quant_t)
        quant = torch.cat([upsample_t, quant_b], 1)
        dec = self.dec(quant)

        return dec

    def _make_layer(self, block, planes, blocks, stride=1, BNmode='BN'):
        downsample = None
        print("####2 BNmode => ", BNmode)
        if stride != 1 or self.inplanes != planes * block.expansion:
            bnl = nn.BatchNorm2d(planes * block.expansion)
            if BNmode == 'IN':
                bnl = nn.InstanceNorm2d(planes * block.expansion)
            if BNmode == 'GN':
                bnl = nn.GroupNorm(32, planes * block.expansion)
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                bnl,
            )
            # Adding AdaIN

        layers = []
        if BNmode == 'GN':
            layers.append(block(self.inplanes, planes, stride, downsample, BNmode=BNmode))
        else:
            layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            if BNmode == 'GN':
                layers.append(block(self.inplanes, planes, BNmode=BNmode))
            else:
                layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    @staticmethod
    def _gaussian_filter(channel, kernel_size, sigma=1.5):
        gauss = torch.Tensor(
            [math.exp(-(x - kernel_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(kernel_size)])
        gauss_1D = (gauss / gauss.sum()).unsqueeze(1)
        gauss_2D = gauss_1D.mm(gauss_1D.t()).float().unsqueeze(0).unsqueeze(0)
        return gauss_2D.expand(channel, 1, kernel_size, kernel_size).contiguous()

    def my_cosine_sync(self, x1, x2, x3, kernel_size=3, sigma=1.5):
        nb, ch, row, col = x1.size()
        cosine_1 = F.cosine_similarity(x1, x2, dim=1).unsqueeze(1)
        cosine_2 = F.cosine_similarity(x1, x3, dim=1).unsqueeze(1)
        cosine = cosine_1 + cosine_2
        filters = self._gaussian_filter(1, kernel_size, sigma).to(cosine.device)
        attenMap = F.conv2d(cosine, filters, padding=1)
        attenMap = attenMap.expand(nb, ch, row, col).contiguous()
        x2 = x2 * attenMap
        x3 = x3 * attenMap
        return x1, x2, x3

    def _status_sync(self, x1, x2, kernel_size=3, sigma=1.5):
        nb, ch, row, col = x1.size()
        cosine = F.cosine_similarity(x1, x2, dim=1).unsqueeze(1)
        filters = self._gaussian_filter(1, kernel_size, sigma).to(cosine.device)
        # print("Cosine =>", cosine.size(), "Filter =>", filters.size())
        attenMap = F.conv2d(cosine, filters, padding=1)
        attenMap = attenMap.expand(nb, ch, row, col).contiguous()
        x1 = x1 * attenMap
        x2 = x2 * attenMap
        return x1, x2

    def forward(self, x):
        if self.src_ch == 1:
            x = torch.cat([x, x, x], dim=1)
        # forward

        enc_t, quant_t, quant_b, diff, _, _ = self.encode(x)
        dec = self.decode(quant_t, quant_b)

        x1 = self.stem_conv(x)
        x1 = self.stem_norm(x1)
        x2 = self.relu(x1)

        # Encoder
        x3 = self.layer1(x2)
        x4 = self.layer2(x3)
        x5 = self.layer3(x4)
        x5_cat = torch.cat([x5, quant_t], dim=1)
        x6 = self.layer4(x5_cat)

        ## stage 1
        x7_L = self.deconvL1(x6)
        x7_a = self.deconva1(x6)
        x7_b = self.deconvb1(x6)
        # sync
        x7_L, x7_a, x7_b = self.my_cosine_sync(x7_L, x7_a, x7_b)

        x7_L = x7_L + x5
        x7_a = x7_a + x5
        x7_b = x7_b + x5

        ## stage 2
        x8_L = self.upResL1(x7_L)
        x8_L = self.deconvL2(x8_L)
        x8_a = self.upResa1(x7_a)
        x8_a = self.deconva2(x8_a)
        x8_b = self.upResb1(x7_b)
        x8_b = self.deconvb2(x8_b)
        # sync
        x8_L, x8_a, x8_b = self.my_cosine_sync(x8_L, x8_a, x8_b)

        x8_L = x8_L + x4
        x8_a = x8_a + x4
        x8_b = x8_b + x4

        ## stage 3
        x9_L = self.upResL2(x8_L)
        x9_L = self.deconvL3(x9_L)
        x9_a = self.upResa2(x8_a)
        x9_a = self.deconva3(x9_a)
        x9_b = self.upResb2(x8_b)
        x9_b = self.deconvb3(x9_b)

        # sync
        x9_L, x9_a, x9_b = self.my_cosine_sync(x9_L, x9_a, x9_b)

        x9_L = x9_L + x3
        x9_a = x9_a + x3
        x9_b = x9_b + x3

        ## stage 4
        x10_L = self.upResL3(x9_L)
        x10_L = self.deconvL4(x10_L)
        x10_a = self.upResa3(x9_a)
        x10_a = self.deconva4(x10_a)
        x10_b = self.upResb3(x9_b)
        x10_b = self.deconvb4(x10_b)
        # sync
        x10_L, x10_a, x10_b = self.my_cosine_sync(x10_L, x10_a, x10_b)

        ## Final stage
        x_final_L = self.predL(x10_L)
        x_final_a = self.preda(x10_a)
        x_final_b = self.predb(x10_b)

        return x_final_L, x_final_a, x_final_b, dec, diff


class ResNet_pix2pixHD_Cannyedge2(nn.Module):

    def __init__(self, block, layers, src_ch, tar_ch, BNmode,
                 in_channel=3, channel=128, n_res_block=2, n_res_channel=32, embed_dim=256, n_embed=512):
        super(ResNet_pix2pixHD_Cannyedge2, self).__init__()

        self.edge_detection = CannyFilter(use_cuda=True)
        self.enc_b = Encoder(in_channel, channel, n_res_block, n_res_channel, stride=4)
        self.enc_t = Encoder(channel, channel, n_res_block, n_res_channel, stride=2)
        self.quantize_conv_t = nn.Conv2d(channel, embed_dim, 1)
        self.quantize_t = Quantize(embed_dim, n_embed)
        self.dec_t = Decoder(
            embed_dim, embed_dim, channel, n_res_block, n_res_channel, stride=2
        )
        self.quantize_conv_b = nn.Conv2d(embed_dim + channel, embed_dim, 1)
        self.quantize_b = Quantize(embed_dim, n_embed)
        self.upsample_t = nn.ConvTranspose2d(
            embed_dim, embed_dim, 4, stride=2, padding=1
        )
        self.dec = Decoder(
            embed_dim + embed_dim,
            in_channel,
            channel,
            n_res_block,
            n_res_channel,
            stride=4,
        )

        kernels = [64, 128, 256, 512]

        self.src_ch = src_ch
        self.inplanes = kernels[0]
        self.stem_conv = nn.Conv2d(3, kernels[0], kernel_size=7,
                                   stride=2, padding=3, bias=False)

        self.stem_conv_edge = nn.Conv2d(1, kernels[0], kernel_size=7,
                                        stride=2, padding=3, bias=False)
        self.stem_norm_edge = nn.InstanceNorm2d(kernels[0])
        self.relu_edge = nn.ReLU(inplace=True)

        print("####1 BNmode => ", BNmode)
        if BNmode == 'BN':
            self.stem_norm = nn.BatchNorm2d(kernels[0])
        elif BNmode == 'IN':
            self.stem_norm = nn.InstanceNorm2d(kernels[0])
        elif BNmode == 'GN':
            self.stem_norm = nn.GroupNorm(32, kernels[0])
        # Adding AdaIN, combined BN and IN
        self.relu = nn.ReLU(inplace=True)
        self.inplanes = kernels[0] + kernels[0]
        self.layer1 = self._make_layer(block, kernels[0], layers[0], stride=1, BNmode=BNmode)
        self.layer2 = self._make_layer(block, kernels[1], layers[1], stride=2, BNmode=BNmode)
        self.layer3 = self._make_layer(block, kernels[2], layers[2], stride=2, BNmode=BNmode)
        self.inplanes = kernels[2] + kernels[2]
        self.layer4 = self._make_layer(block, kernels[3], layers[3], stride=2, BNmode=BNmode)

        # L components
        self.deconvL1 = upconv(kernels[3], kernels[2], ratio="x2")
        self.inplanes = kernels[2]
        self.upResL1 = self._make_layer(block, kernels[2], layers[2], BNmode=BNmode)
        self.deconvL2 = upconv(kernels[2], kernels[1], ratio="x2")
        self.inplanes = kernels[1]
        self.upResL2 = self._make_layer(block, kernels[1], layers[1], BNmode=BNmode)
        self.deconvL3 = upconv(kernels[1], kernels[0], ratio="x2")
        self.inplanes = kernels[0]
        self.upResL3 = self._make_layer(block, kernels[0], layers[0], BNmode=BNmode)
        self.deconvL4 = upconv(kernels[0], kernels[0], ratio="x2")
        self.predL = nn.Conv2d(kernels[0], tar_ch[0], kernel_size=3, stride=1, padding=1, bias=False)

        # a components
        self.deconva1 = upconv(kernels[3], kernels[2], ratio="x2")
        self.inplanes = kernels[2]
        self.upResa1 = self._make_layer(block, kernels[2], layers[2], BNmode=BNmode)
        self.deconva2 = upconv(kernels[2], kernels[1], ratio="x2")
        self.inplanes = kernels[1]
        self.upResa2 = self._make_layer(block, kernels[1], layers[1], BNmode=BNmode)
        self.deconva3 = upconv(kernels[1], kernels[0], ratio="x2")
        self.inplanes = kernels[0]
        self.upResa3 = self._make_layer(block, kernels[0], layers[0], BNmode=BNmode)
        self.deconva4 = upconv(kernels[0], kernels[0], ratio="x2")
        self.preda = nn.Conv2d(kernels[0], tar_ch[1], kernel_size=3, stride=1, padding=1, bias=False)

        # b components
        self.deconvb1 = upconv(kernels[3], kernels[2], ratio="x2")
        self.inplanes = kernels[2]
        self.upResb1 = self._make_layer(block, kernels[2], layers[2], BNmode=BNmode)
        self.deconvb2 = upconv(kernels[2], kernels[1], ratio="x2")
        self.inplanes = kernels[1]
        self.upResb2 = self._make_layer(block, kernels[1], layers[1], BNmode=BNmode)
        self.deconvb3 = upconv(kernels[1], kernels[0], ratio="x2")
        self.inplanes = kernels[0]
        self.upResb3 = self._make_layer(block, kernels[0], layers[0], BNmode=BNmode)
        self.deconvb4 = upconv(kernels[0], kernels[0], ratio="x2")
        self.predb = nn.Conv2d(kernels[0], tar_ch[2], kernel_size=3, stride=1, padding=1, bias=False)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def encode(self, input):
        enc_b = self.enc_b(input)
        enc_t = self.enc_t(enc_b)
        quant_t = self.quantize_conv_t(enc_t).permute(0, 2, 3, 1)
        quant_t, diff_t, id_t = self.quantize_t(quant_t)
        quant_t = quant_t.permute(0, 3, 1, 2)
        diff_t = diff_t.unsqueeze(0)

        dec_t = self.dec_t(quant_t)
        enc_b = torch.cat([dec_t, enc_b], 1)

        quant_b = self.quantize_conv_b(enc_b).permute(0, 2, 3, 1)
        quant_b, diff_b, id_b = self.quantize_b(quant_b)
        quant_b = quant_b.permute(0, 3, 1, 2)
        diff_b = diff_b.unsqueeze(0)

        return enc_t, quant_t, quant_b, diff_t + diff_b, id_t, id_b

    def decode(self, quant_t, quant_b):
        upsample_t = self.upsample_t(quant_t)
        quant = torch.cat([upsample_t, quant_b], 1)
        dec = self.dec(quant)

        return dec

    def _make_layer(self, block, planes, blocks, stride=1, BNmode='BN'):
        downsample = None
        print("####2 BNmode => ", BNmode)
        if stride != 1 or self.inplanes != planes * block.expansion:
            bnl = nn.BatchNorm2d(planes * block.expansion)
            if BNmode == 'IN':
                bnl = nn.InstanceNorm2d(planes * block.expansion)
            if BNmode == 'GN':
                bnl = nn.GroupNorm(32, planes * block.expansion)
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                bnl,
            )
            # Adding AdaIN

        layers = []
        if BNmode == 'GN':
            layers.append(block(self.inplanes, planes, stride, downsample, BNmode=BNmode))
        else:
            layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            if BNmode == 'GN':
                layers.append(block(self.inplanes, planes, BNmode=BNmode))
            else:
                layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    @staticmethod
    def _gaussian_filter(channel, kernel_size, sigma=1.5):
        gauss = torch.Tensor(
            [math.exp(-(x - kernel_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(kernel_size)])
        gauss_1D = (gauss / gauss.sum()).unsqueeze(1)
        gauss_2D = gauss_1D.mm(gauss_1D.t()).float().unsqueeze(0).unsqueeze(0)
        return gauss_2D.expand(channel, 1, kernel_size, kernel_size).contiguous()

    def my_cosine_sync(self, x1, x2, x3, kernel_size=3, sigma=1.5):
        nb, ch, row, col = x1.size()
        cosine_1 = F.cosine_similarity(x1, x2, dim=1).unsqueeze(1)
        cosine_2 = F.cosine_similarity(x1, x3, dim=1).unsqueeze(1)
        cosine = cosine_1 + cosine_2
        filters = self._gaussian_filter(1, kernel_size, sigma).to(cosine.device)
        attenMap = F.conv2d(cosine, filters, padding=1)
        attenMap = attenMap.expand(nb, ch, row, col).contiguous()
        x2 = x2 * attenMap
        x3 = x3 * attenMap
        return x1, x2, x3

    def _status_sync(self, x1, x2, kernel_size=3, sigma=1.5):
        nb, ch, row, col = x1.size()
        cosine = F.cosine_similarity(x1, x2, dim=1).unsqueeze(1)
        filters = self._gaussian_filter(1, kernel_size, sigma).to(cosine.device)
        # print("Cosine =>", cosine.size(), "Filter =>", filters.size())
        attenMap = F.conv2d(cosine, filters, padding=1)
        attenMap = attenMap.expand(nb, ch, row, col).contiguous()
        x1 = x1 * attenMap
        x2 = x2 * attenMap
        return x1, x2

    def forward(self, x):
        if self.src_ch == 1:
            x = torch.cat([x, x, x], dim=1)
        # forward

        _, _, _, _, _, thin_edges = self.edge_detection(x)

        enc_t, quant_t, quant_b, diff, _, _ = self.encode(x)
        dec = self.decode(quant_t, quant_b)

        # edge parts
        canny_edge = self.stem_conv_edge(thin_edges)
        canny_edge = self.stem_norm_edge(canny_edge)
        canny_edge = self.relu_edge(canny_edge)

        x1 = self.stem_conv(x)
        x1 = self.stem_norm(x1)
        x2 = self.relu(x1)

        # Encoder
        x2_cat = torch.cat([x2, canny_edge], dim=1)
        x3 = self.layer1(x2_cat)
        x4 = self.layer2(x3)
        x5 = self.layer3(x4)
        x5_cat = torch.cat([x5, quant_t], dim=1)
        x6 = self.layer4(x5_cat)

        ## stage 1
        x7_L = self.deconvL1(x6)
        x7_a = self.deconva1(x6)
        x7_b = self.deconvb1(x6)
        # sync
        x7_L, x7_a, x7_b = self.my_cosine_sync(x7_L, x7_a, x7_b)

        x7_L = x7_L + x5
        x7_a = x7_a + x5
        x7_b = x7_b + x5

        ## stage 2
        x8_L = self.upResL1(x7_L)
        x8_L = self.deconvL2(x8_L)
        x8_a = self.upResa1(x7_a)
        x8_a = self.deconva2(x8_a)
        x8_b = self.upResb1(x7_b)
        x8_b = self.deconvb2(x8_b)
        # sync
        x8_L, x8_a, x8_b = self.my_cosine_sync(x8_L, x8_a, x8_b)

        x8_L = x8_L + x4
        x8_a = x8_a + x4
        x8_b = x8_b + x4

        ## stage 3
        x9_L = self.upResL2(x8_L)
        x9_L = self.deconvL3(x9_L)
        x9_a = self.upResa2(x8_a)
        x9_a = self.deconva3(x9_a)
        x9_b = self.upResb2(x8_b)
        x9_b = self.deconvb3(x9_b)

        # sync
        x9_L, x9_a, x9_b = self.my_cosine_sync(x9_L, x9_a, x9_b)

        x9_L = x9_L + x3
        x9_a = x9_a + x3
        x9_b = x9_b + x3
        # edge
        # x9_L = torch.cat([x9_L, canny_edge], dim=1)
        ## stage 4
        x10_L = self.upResL3(x9_L)
        x10_L = self.deconvL4(x10_L)
        x10_a = self.upResa3(x9_a)
        x10_a = self.deconva4(x10_a)
        x10_b = self.upResb3(x9_b)
        x10_b = self.deconvb4(x10_b)
        # sync
        x10_L, x10_a, x10_b = self.my_cosine_sync(x10_L, x10_a, x10_b)

        ## Final stage
        x_final_L = self.predL(x10_L)
        x_final_a = self.preda(x10_a)
        x_final_b = self.predb(x10_b)

        return x_final_L, x_final_a, x_final_b, dec, diff


class ResNet_pix2pixHD_input(nn.Module):

    def __init__(self, block, layers, src_ch, tar_ch, BNmode):
        super(ResNet_pix2pixHD_input, self).__init__()

        kernels = [16, 32, 64, 128]

        self.src_ch = src_ch
        self.inplanes = kernels[0]
        self.stem_conv = nn.Conv2d(3, kernels[0], kernel_size=7,
                                   stride=2, padding=3, bias=False)

        print("####1 BNmode => ", BNmode)
        if BNmode == 'BN':
            self.stem_norm = nn.BatchNorm2d(kernels[0])
        elif BNmode == 'IN':
            self.stem_norm = nn.InstanceNorm2d(kernels[0])
        elif BNmode == 'GN':
            self.stem_norm = nn.GroupNorm(32, kernels[0])
        # Adding AdaIN, combined BN and IN
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(block, kernels[0], layers[0], stride=1, BNmode=BNmode)
        self.layer2 = self._make_layer(block, kernels[1], layers[1], stride=2, BNmode=BNmode)
        self.layer3 = self._make_layer(block, kernels[2], layers[2], stride=2, BNmode=BNmode)
        self.layer4 = self._make_layer(block, kernels[3], layers[3], stride=2, BNmode=BNmode)

        # L components
        self.deconvL1 = upconv(kernels[3], kernels[2], ratio="x2")
        self.inplanes = kernels[2]
        self.upResL1 = self._make_layer(block, kernels[2], layers[2], BNmode=BNmode)
        self.deconvL2 = upconv(kernels[2], kernels[1], ratio="x2")
        self.inplanes = kernels[1]
        self.upResL2 = self._make_layer(block, kernels[1], layers[1], BNmode=BNmode)
        self.deconvL3 = upconv(kernels[1], kernels[0], ratio="x2")
        self.inplanes = kernels[0]
        self.upResL3 = self._make_layer(block, kernels[0], layers[0], BNmode=BNmode)
        self.deconvL4 = upconv(kernels[0], kernels[0], ratio="x2")
        self.predL = nn.Conv2d(kernels[0], tar_ch[0], kernel_size=3, stride=1, padding=1, bias=False)

        # a components
        self.deconva1 = upconv(kernels[3], kernels[2], ratio="x2")
        self.inplanes = kernels[2]
        self.upResa1 = self._make_layer(block, kernels[2], layers[2], BNmode=BNmode)
        self.deconva2 = upconv(kernels[2], kernels[1], ratio="x2")
        self.inplanes = kernels[1]
        self.upResa2 = self._make_layer(block, kernels[1], layers[1], BNmode=BNmode)
        self.deconva3 = upconv(kernels[1], kernels[0], ratio="x2")
        self.inplanes = kernels[0]
        self.upResa3 = self._make_layer(block, kernels[0], layers[0], BNmode=BNmode)
        self.deconva4 = upconv(kernels[0], kernels[0], ratio="x2")
        self.preda = nn.Conv2d(kernels[0], tar_ch[1], kernel_size=3, stride=1, padding=1, bias=False)

        # b components
        self.deconvb1 = upconv(kernels[3], kernels[2], ratio="x2")
        self.inplanes = kernels[2]
        self.upResb1 = self._make_layer(block, kernels[2], layers[2], BNmode=BNmode)
        self.deconvb2 = upconv(kernels[2], kernels[1], ratio="x2")
        self.inplanes = kernels[1]
        self.upResb2 = self._make_layer(block, kernels[1], layers[1], BNmode=BNmode)
        self.deconvb3 = upconv(kernels[1], kernels[0], ratio="x2")
        self.inplanes = kernels[0]
        self.upResb3 = self._make_layer(block, kernels[0], layers[0], BNmode=BNmode)
        self.deconvb4 = upconv(kernels[0], kernels[0], ratio="x2")
        self.predb = nn.Conv2d(kernels[0], tar_ch[2], kernel_size=3, stride=1, padding=1, bias=False)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1, BNmode='BN'):
        downsample = None
        print("####2 BNmode => ", BNmode)
        if stride != 1 or self.inplanes != planes * block.expansion:
            bnl = nn.BatchNorm2d(planes * block.expansion)
            if BNmode == 'IN':
                bnl = nn.InstanceNorm2d(planes * block.expansion)
            if BNmode == 'GN':
                bnl = nn.GroupNorm(32, planes * block.expansion)
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                bnl,
            )
            # Adding AdaIN

        layers = []
        if BNmode == 'GN':
            layers.append(block(self.inplanes, planes, stride, downsample, BNmode=BNmode))
        else:
            layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            if BNmode == 'GN':
                layers.append(block(self.inplanes, planes, BNmode=BNmode))
            else:
                layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    @staticmethod
    def _gaussian_filter(channel, kernel_size, sigma=1.5):
        gauss = torch.Tensor(
            [math.exp(-(x - kernel_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(kernel_size)])
        gauss_1D = (gauss / gauss.sum()).unsqueeze(1)
        gauss_2D = gauss_1D.mm(gauss_1D.t()).float().unsqueeze(0).unsqueeze(0)
        return gauss_2D.expand(channel, 1, kernel_size, kernel_size).contiguous()

    def my_cosine_sync(self, x1, x2, x3, kernel_size=3, sigma=1.5):
        nb, ch, row, col = x1.size()
        cosine_1 = F.cosine_similarity(x1, x2, dim=1).unsqueeze(1)
        cosine_2 = F.cosine_similarity(x1, x3, dim=1).unsqueeze(1)
        cosine = cosine_1 + cosine_2
        filters = self._gaussian_filter(1, kernel_size, sigma).to(cosine.device)
        attenMap = F.conv2d(cosine, filters, padding=1)
        attenMap = attenMap.expand(nb, ch, row, col).contiguous()
        x2 = x2 * attenMap
        x3 = x3 * attenMap
        return x1, x2, x3

    def _status_sync(self, x1, x2, kernel_size=3, sigma=1.5):
        nb, ch, row, col = x1.size()
        cosine = F.cosine_similarity(x1, x2, dim=1).unsqueeze(1)
        filters = self._gaussian_filter(1, kernel_size, sigma).to(cosine.device)
        # print("Cosine =>", cosine.size(), "Filter =>", filters.size())
        attenMap = F.conv2d(cosine, filters, padding=1)
        attenMap = attenMap.expand(nb, ch, row, col).contiguous()
        x1 = x1 * attenMap
        x2 = x2 * attenMap
        return x1, x2

    def forward(self, x):
        if self.src_ch == 1:
            x = torch.cat([x, x, x], dim=1)
        # forward

        x1 = self.stem_conv(x)
        x1 = self.stem_norm(x1)
        x2 = self.relu(x1)

        # Encoder
        x3 = self.layer1(x2)
        x4 = self.layer2(x3)
        x5 = self.layer3(x4)
        x6 = self.layer4(x5)

        ## stage 1
        x7_L = self.deconvL1(x6)
        x7_a = self.deconva1(x6)
        x7_b = self.deconvb1(x6)
        # sync
        x7_L, x7_a, x7_b = self.my_cosine_sync(x7_L, x7_a, x7_b)

        x7_L = x7_L + x5
        x7_a = x7_a + x5
        x7_b = x7_b + x5

        ## stage 2
        x8_L = self.upResL1(x7_L)
        x8_L = self.deconvL2(x8_L)
        x8_a = self.upResa1(x7_a)
        x8_a = self.deconva2(x8_a)
        x8_b = self.upResb1(x7_b)
        x8_b = self.deconvb2(x8_b)
        # sync
        x8_L, x8_a, x8_b = self.my_cosine_sync(x8_L, x8_a, x8_b)

        x8_L = x8_L + x4
        x8_a = x8_a + x4
        x8_b = x8_b + x4

        ## stage 3
        x9_L = self.upResL2(x8_L)
        x9_L = self.deconvL3(x9_L)
        x9_a = self.upResa2(x8_a)
        x9_a = self.deconva3(x9_a)
        x9_b = self.upResb2(x8_b)
        x9_b = self.deconvb3(x9_b)

        # sync
        x9_L, x9_a, x9_b = self.my_cosine_sync(x9_L, x9_a, x9_b)

        x9_L = x9_L + x3
        x9_a = x9_a + x3
        x9_b = x9_b + x3

        ## stage 4
        x10_L = self.upResL3(x9_L)
        x10_L = self.deconvL4(x10_L)
        x10_L_temp = x10_L
        x10_a = self.upResa3(x9_a)
        x10_a = self.deconva4(x10_a)
        x10_a_temp = x10_a
        x10_b = self.upResb3(x9_b)
        x10_b = self.deconvb4(x10_b)
        x10_b_temp = x10_b
        # sync
        x10_L, x10_a, x10_b = self.my_cosine_sync(x10_L, x10_a, x10_b)

        ## Final stage
        x_final_L = self.predL(x10_L)
        x_final_a = self.preda(x10_a)
        x_final_b = self.predb(x10_b)

        return x_final_L, x_final_a, x_final_b, x10_L_temp, x10_a_temp, x10_b_temp


class ResNet_pix2pixHD_edgeparts(nn.Module):

    def __init__(self, block, layers, src_ch, tar_ch, BNmode):
        super(ResNet_pix2pixHD_edgeparts, self).__init__()

        self.edge_detection = CannyFilter(use_cuda=True)

        kernels = [16, 32, 64, 128]

        self.src_ch = src_ch
        self.inplanes = kernels[0]
        self.stem_conv_edge = nn.Conv2d(3, kernels[0], kernel_size=7,
                                        stride=2, padding=3, bias=False)
        self.stem_norm_edge = nn.InstanceNorm2d(kernels[0])
        self.relu_edge = nn.ReLU(inplace=True)

        print("####1 BNmode => ", BNmode)
        if BNmode == 'BN':
            self.stem_norm = nn.BatchNorm2d(kernels[0])
        elif BNmode == 'IN':
            self.stem_norm = nn.InstanceNorm2d(kernels[0])
        elif BNmode == 'GN':
            self.stem_norm = nn.GroupNorm(32, kernels[0])
        # Adding AdaIN, combined BN and IN
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, kernels[0], layers[0], stride=1, BNmode=BNmode)
        self.layer2 = self._make_layer(block, kernels[1], layers[1], stride=2, BNmode=BNmode)
        self.layer3 = self._make_layer(block, kernels[2], layers[2], stride=2, BNmode=BNmode)
        self.layer4 = self._make_layer(block, kernels[3], layers[3], stride=2, BNmode=BNmode)

        # L components
        self.deconvL1 = upconv(kernels[3], kernels[2], ratio="x2")
        self.inplanes = kernels[2]
        self.upResL1 = self._make_layer(block, kernels[2], layers[2], BNmode=BNmode)
        self.deconvL2 = upconv(kernels[2], kernels[1], ratio="x2")
        self.inplanes = kernels[1]
        self.upResL2 = self._make_layer(block, kernels[1], layers[1], BNmode=BNmode)
        self.deconvL3 = upconv(kernels[1], kernels[0], ratio="x2")
        self.inplanes = kernels[0]
        self.upResL3 = self._make_layer(block, kernels[0], layers[0], BNmode=BNmode)
        self.deconvL4 = upconv(kernels[0], kernels[0], ratio="x2")
        self.predL = nn.Conv2d(kernels[0], tar_ch[0], kernel_size=3, stride=1, padding=1, bias=False)

        # a components
        self.deconva1 = upconv(kernels[3], kernels[2], ratio="x2")
        self.inplanes = kernels[2]
        self.upResa1 = self._make_layer(block, kernels[2], layers[2], BNmode=BNmode)
        self.deconva2 = upconv(kernels[2], kernels[1], ratio="x2")
        self.inplanes = kernels[1]
        self.upResa2 = self._make_layer(block, kernels[1], layers[1], BNmode=BNmode)
        self.deconva3 = upconv(kernels[1], kernels[0], ratio="x2")
        self.inplanes = kernels[0]
        self.upResa3 = self._make_layer(block, kernels[0], layers[0], BNmode=BNmode)
        self.deconva4 = upconv(kernels[0], kernels[0], ratio="x2")
        self.preda = nn.Conv2d(kernels[0], tar_ch[1], kernel_size=3, stride=1, padding=1, bias=False)

        # b components
        self.deconvb1 = upconv(kernels[3], kernels[2], ratio="x2")
        self.inplanes = kernels[2]
        self.upResb1 = self._make_layer(block, kernels[2], layers[2], BNmode=BNmode)
        self.deconvb2 = upconv(kernels[2], kernels[1], ratio="x2")
        self.inplanes = kernels[1]
        self.upResb2 = self._make_layer(block, kernels[1], layers[1], BNmode=BNmode)
        self.deconvb3 = upconv(kernels[1], kernels[0], ratio="x2")
        self.inplanes = kernels[0]
        self.upResb3 = self._make_layer(block, kernels[0], layers[0], BNmode=BNmode)
        self.deconvb4 = upconv(kernels[0], kernels[0], ratio="x2")
        self.predb = nn.Conv2d(kernels[0], tar_ch[2], kernel_size=3, stride=1, padding=1, bias=False)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1, BNmode='BN'):
        downsample = None
        print("####2 BNmode => ", BNmode)
        if stride != 1 or self.inplanes != planes * block.expansion:
            bnl = nn.BatchNorm2d(planes * block.expansion)
            if BNmode == 'IN':
                bnl = nn.InstanceNorm2d(planes * block.expansion)
            if BNmode == 'GN':
                bnl = nn.GroupNorm(32, planes * block.expansion)
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                bnl,
            )
            # Adding AdaIN

        layers = []
        if BNmode == 'GN':
            layers.append(block(self.inplanes, planes, stride, downsample, BNmode=BNmode))
        else:
            layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            if BNmode == 'GN':
                layers.append(block(self.inplanes, planes, BNmode=BNmode))
            else:
                layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    @staticmethod
    def _gaussian_filter(channel, kernel_size, sigma=1.5):
        gauss = torch.Tensor(
            [math.exp(-(x - kernel_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(kernel_size)])
        gauss_1D = (gauss / gauss.sum()).unsqueeze(1)
        gauss_2D = gauss_1D.mm(gauss_1D.t()).float().unsqueeze(0).unsqueeze(0)
        return gauss_2D.expand(channel, 1, kernel_size, kernel_size).contiguous()

    def my_cosine_sync(self, x1, x2, x3, kernel_size=3, sigma=1.5):
        nb, ch, row, col = x1.size()
        cosine_1 = F.cosine_similarity(x1, x2, dim=1).unsqueeze(1)
        cosine_2 = F.cosine_similarity(x1, x3, dim=1).unsqueeze(1)
        cosine = cosine_1 + cosine_2
        filters = self._gaussian_filter(1, kernel_size, sigma).to(cosine.device)
        attenMap = F.conv2d(cosine, filters, padding=1)
        attenMap = attenMap.expand(nb, ch, row, col).contiguous()
        x2 = x2 * attenMap
        x3 = x3 * attenMap
        return x1, x2, x3

    def _status_sync(self, x1, x2, kernel_size=3, sigma=1.5):
        nb, ch, row, col = x1.size()
        cosine = F.cosine_similarity(x1, x2, dim=1).unsqueeze(1)
        filters = self._gaussian_filter(1, kernel_size, sigma).to(cosine.device)
        # print("Cosine =>", cosine.size(), "Filter =>", filters.size())
        attenMap = F.conv2d(cosine, filters, padding=1)
        attenMap = attenMap.expand(nb, ch, row, col).contiguous()
        x1 = x1 * attenMap
        x2 = x2 * attenMap
        return x1, x2

    def forward(self, x):
        if self.src_ch == 1:
            x = torch.cat([x, x, x], dim=1)
        # forward

        _, _, _, _, _, thin_edges = self.edge_detection(x)
        thin_edges = torch.cat([thin_edges, thin_edges, thin_edges], dim=1)

        # edge parts

        x1 = self.stem_conv_edge(thin_edges)
        x1 = self.stem_norm_edge(x1)
        x2 = self.relu_edge(x1)

        # Encoder

        x3 = self.layer1(x2)
        x4 = self.layer2(x3)
        x5 = self.layer3(x4)
        x6 = self.layer4(x5)

        ## stage 1
        x7_L = self.deconvL1(x6)
        x7_a = self.deconva1(x6)
        x7_b = self.deconvb1(x6)
        # sync
        x7_L, x7_a, x7_b = self.my_cosine_sync(x7_L, x7_a, x7_b)

        x7_L = x7_L + x5
        x7_a = x7_a + x5
        x7_b = x7_b + x5

        ## stage 2
        x8_L = self.upResL1(x7_L)
        x8_L = self.deconvL2(x8_L)
        x8_a = self.upResa1(x7_a)
        x8_a = self.deconva2(x8_a)
        x8_b = self.upResb1(x7_b)
        x8_b = self.deconvb2(x8_b)
        # sync
        x8_L, x8_a, x8_b = self.my_cosine_sync(x8_L, x8_a, x8_b)

        x8_L = x8_L + x4
        x8_a = x8_a + x4
        x8_b = x8_b + x4

        ## stage 3
        x9_L = self.upResL2(x8_L)
        x9_L = self.deconvL3(x9_L)
        x9_a = self.upResa2(x8_a)
        x9_a = self.deconva3(x9_a)
        x9_b = self.upResb2(x8_b)
        x9_b = self.deconvb3(x9_b)

        # sync
        x9_L, x9_a, x9_b = self.my_cosine_sync(x9_L, x9_a, x9_b)

        x9_L = x9_L + x3
        x9_a = x9_a + x3
        x9_b = x9_b + x3

        ## stage 4
        x10_L = self.upResL3(x9_L)
        x10_L = self.deconvL4(x10_L)
        x10_L_temp = x10_L
        x10_a = self.upResa3(x9_a)
        x10_a = self.deconva4(x10_a)
        x10_a_temp = x10_a
        x10_b = self.upResb3(x9_b)
        x10_b = self.deconvb4(x10_b)
        x10_b_temp = x10_b
        # sync
        x10_L, x10_a, x10_b = self.my_cosine_sync(x10_L, x10_a, x10_b)

        ## Final stage
        x_final_L = self.predL(x10_L)
        x_final_a = self.preda(x10_a)
        x_final_b = self.predb(x10_b)

        return x_final_L, x_final_a, x_final_b, x10_L_temp, x10_a_temp, x10_b_temp


class ResNet_pix2pixHD_final(nn.Module):

    def __init__(self, block, layers, src_ch, tar_ch, BNmode):
        super(ResNet_pix2pixHD_final, self).__init__()

        kernels = [128, 256, 256, 512]

        self.src_ch = src_ch
        self.inplanes = kernels[0]
        self.stem_conv = nn.Conv2d(96, kernels[0], kernel_size=7,
                                   stride=2, padding=3, bias=False)
        print("####1 BNmode => ", BNmode)
        if BNmode == 'BN':
            self.stem_norm = nn.BatchNorm2d(kernels[0])
        elif BNmode == 'IN':
            self.stem_norm = nn.InstanceNorm2d(kernels[0])
        elif BNmode == 'GN':
            self.stem_norm = nn.GroupNorm(64, kernels[0])
        # Adding AdaIN, combined BN and IN
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(block, kernels[0], layers[0], stride=1, BNmode=BNmode)
        self.layer2 = self._make_layer(block, kernels[1], layers[1], stride=2, BNmode=BNmode)
        self.layer3 = self._make_layer(block, kernels[2], layers[2], stride=2, BNmode=BNmode)
        self.layer4 = self._make_layer(block, kernels[3], layers[3], stride=2, BNmode=BNmode)

        # L components
        self.deconvL1 = upconv(kernels[3], kernels[2], ratio="x2")
        self.inplanes = kernels[2]
        self.upResL1 = self._make_layer(block, kernels[2], layers[2], BNmode=BNmode)
        self.deconvL2 = upconv(kernels[2], kernels[1], ratio="x2")
        self.inplanes = kernels[1]
        self.upResL2 = self._make_layer(block, kernels[1], layers[1], BNmode=BNmode)
        self.deconvL3 = upconv(kernels[1], kernels[0], ratio="x2")
        self.inplanes = kernels[0]
        self.upResL3 = self._make_layer(block, kernels[0], layers[0], BNmode=BNmode)
        self.deconvL4 = upconv(kernels[0], kernels[0], ratio="x2")
        self.predL = nn.Conv2d(kernels[0], tar_ch[0], kernel_size=3, stride=1, padding=1, bias=False)

        # a components
        self.deconva1 = upconv(kernels[3], kernels[2], ratio="x2")
        self.inplanes = kernels[2]
        self.upResa1 = self._make_layer(block, kernels[2], layers[2], BNmode=BNmode)
        self.deconva2 = upconv(kernels[2], kernels[1], ratio="x2")
        self.inplanes = kernels[1]
        self.upResa2 = self._make_layer(block, kernels[1], layers[1], BNmode=BNmode)
        self.deconva3 = upconv(kernels[1], kernels[0], ratio="x2")
        self.inplanes = kernels[0]
        self.upResa3 = self._make_layer(block, kernels[0], layers[0], BNmode=BNmode)
        self.deconva4 = upconv(kernels[0], kernels[0], ratio="x2")
        self.preda = nn.Conv2d(kernels[0], tar_ch[1], kernel_size=3, stride=1, padding=1, bias=False)

        # b components
        self.deconvb1 = upconv(kernels[3], kernels[2], ratio="x2")
        self.inplanes = kernels[2]
        self.upResb1 = self._make_layer(block, kernels[2], layers[2], BNmode=BNmode)
        self.deconvb2 = upconv(kernels[2], kernels[1], ratio="x2")
        self.inplanes = kernels[1]
        self.upResb2 = self._make_layer(block, kernels[1], layers[1], BNmode=BNmode)
        self.deconvb3 = upconv(kernels[1], kernels[0], ratio="x2")
        self.inplanes = kernels[0]
        self.upResb3 = self._make_layer(block, kernels[0], layers[0], BNmode=BNmode)
        self.deconvb4 = upconv(kernels[0], kernels[0], ratio="x2")
        self.predb = nn.Conv2d(kernels[0], tar_ch[2], kernel_size=3, stride=1, padding=1, bias=False)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1, BNmode='BN'):
        downsample = None
        print("####2 BNmode => ", BNmode)
        if stride != 1 or self.inplanes != planes * block.expansion:
            bnl = nn.BatchNorm2d(planes * block.expansion)
            if BNmode == 'IN':
                bnl = nn.InstanceNorm2d(planes * block.expansion)
            if BNmode == 'GN':
                bnl = nn.GroupNorm(32, planes * block.expansion)
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                bnl,
            )
            # Adding AdaIN

        layers = []
        if BNmode == 'GN':
            layers.append(block(self.inplanes, planes, stride, downsample, BNmode=BNmode))
        else:
            layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            if BNmode == 'GN':
                layers.append(block(self.inplanes, planes, BNmode=BNmode))
            else:
                layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    @staticmethod
    def _gaussian_filter(channel, kernel_size, sigma=1.5):
        gauss = torch.Tensor(
            [math.exp(-(x - kernel_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(kernel_size)])
        gauss_1D = (gauss / gauss.sum()).unsqueeze(1)
        gauss_2D = gauss_1D.mm(gauss_1D.t()).float().unsqueeze(0).unsqueeze(0)
        return gauss_2D.expand(channel, 1, kernel_size, kernel_size).contiguous()

    def my_cosine_sync(self, x1, x2, x3, kernel_size=3, sigma=1.5):
        nb, ch, row, col = x1.size()
        cosine_1 = F.cosine_similarity(x1, x2, dim=1).unsqueeze(1)
        cosine_2 = F.cosine_similarity(x1, x3, dim=1).unsqueeze(1)
        cosine = cosine_1 + cosine_2
        filters = self._gaussian_filter(1, kernel_size, sigma).to(cosine.device)
        attenMap = F.conv2d(cosine, filters, padding=1)
        attenMap = attenMap.expand(nb, ch, row, col).contiguous()
        x2 = x2 * attenMap
        x3 = x3 * attenMap
        return x1, x2, x3

    def _status_sync(self, x1, x2, kernel_size=3, sigma=1.5):
        nb, ch, row, col = x1.size()
        cosine = F.cosine_similarity(x1, x2, dim=1).unsqueeze(1)
        filters = self._gaussian_filter(1, kernel_size, sigma).to(cosine.device)
        # print("Cosine =>", cosine.size(), "Filter =>", filters.size())
        attenMap = F.conv2d(cosine, filters, padding=1)
        attenMap = attenMap.expand(nb, ch, row, col).contiguous()
        x1 = x1 * attenMap
        x2 = x2 * attenMap
        return x1, x2

    def forward(self, x):
        if self.src_ch == 1:
            x = torch.cat([x, x, x], dim=1)
        # forward

        x1 = self.stem_conv(x)
        x1 = self.stem_norm(x1)
        x2 = self.relu(x1)

        # Encoder
        x3 = self.layer1(x2)
        x4 = self.layer2(x3)
        x5 = self.layer3(x4)
        x6 = self.layer4(x5)

        ## stage 1
        x7_L = self.deconvL1(x6)
        x7_a = self.deconva1(x6)
        x7_b = self.deconvb1(x6)
        # sync
        x7_L, x7_a, x7_b = self.my_cosine_sync(x7_L, x7_a, x7_b)

        x7_L = x7_L + x5
        x7_a = x7_a + x5
        x7_b = x7_b + x5

        ## stage 2
        x8_L = self.upResL1(x7_L)
        x8_L = self.deconvL2(x8_L)
        x8_a = self.upResa1(x7_a)
        x8_a = self.deconva2(x8_a)
        x8_b = self.upResb1(x7_b)
        x8_b = self.deconvb2(x8_b)
        # sync
        x8_L, x8_a, x8_b = self.my_cosine_sync(x8_L, x8_a, x8_b)

        x8_L = x8_L + x4
        x8_a = x8_a + x4
        x8_b = x8_b + x4

        ## stage 3
        x9_L = self.upResL2(x8_L)
        x9_L = self.deconvL3(x9_L)
        x9_a = self.upResa2(x8_a)
        x9_a = self.deconva3(x9_a)
        x9_b = self.upResb2(x8_b)
        x9_b = self.deconvb3(x9_b)

        # sync
        x9_L, x9_a, x9_b = self.my_cosine_sync(x9_L, x9_a, x9_b)

        x9_L = x9_L + x3
        x9_a = x9_a + x3
        x9_b = x9_b + x3

        ## stage 4
        x10_L = self.upResL3(x9_L)
        x10_L = self.deconvL4(x10_L)
        x10_a = self.upResa3(x9_a)
        x10_a = self.deconva4(x10_a)
        x10_b = self.upResb3(x9_b)
        x10_b = self.deconvb4(x10_b)
        # sync
        x10_L, x10_a, x10_b = self.my_cosine_sync(x10_L, x10_a, x10_b)

        ## Final stage
        x_final_L = self.predL(x10_L)
        x_final_a = self.preda(x10_a)
        x_final_b = self.predb(x10_b)

        return x_final_L, x_final_a, x_final_b


class ResYNetSync_origin(nn.Module):

    def __init__(self, block, layers, src_ch, tar_ch, BNmode):
        super(ResYNetSync_origin, self).__init__()
        kernels = [64, 128, 256, 512]
        self.src_ch = src_ch
        self.inplanes = kernels[0]
        self.conv1 = nn.Conv2d(3, kernels[0], kernel_size=7,
                               stride=2, padding=3,
                               bias=False)
        #         print("####1 BNmode => ", BNmode)
        if BNmode == 'BN':
            self.bn1 = nn.BatchNorm2d(kernels[0])
        elif BNmode == 'IN':
            self.bn1 = nn.InstanceNorm2d(kernels[0])
        elif BNmode == 'GN':
            self.bn1 = nn.GroupNorm(32, kernels[0])
        self.relu = nn.ReLU(inplace=True)
        # self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=0, ceil_mode=True)

        self.layer1 = self._make_layer(block, kernels[0], layers[0], stride=1, BNmode=BNmode)
        self.layer2 = self._make_layer(block, kernels[1], layers[1], stride=2, BNmode=BNmode)
        self.layer3 = self._make_layer(block, kernels[2], layers[2], stride=2, BNmode=BNmode)
        self.layer4 = self._make_layer(block, kernels[3], layers[3], stride=2, BNmode=BNmode)

        # Left arm
        self.deconvL10 = deconv(kernels[3], kernels[2], ratio="x2")
        self.inplanes = kernels[2]
        self.upResL1 = self._make_layer(block, kernels[2], layers[2], BNmode=BNmode)
        self.deconvL11 = deconv(kernels[2], kernels[1], ratio="x2")
        self.inplanes = kernels[1]
        self.upResL2 = self._make_layer(block, kernels[1], layers[1], BNmode=BNmode)
        self.deconvL12 = deconv(kernels[1], kernels[0], ratio="x2")
        self.inplanes = kernels[0]
        self.upResL3 = self._make_layer(block, kernels[0], layers[0], BNmode=BNmode)
        self.deconvL13 = deconv(kernels[0], kernels[0], ratio="x2")
        self.predL = nn.Conv2d(kernels[0], tar_ch[0], kernel_size=3, stride=1, padding=1, bias=False)

        # Right arm
        self.deconvR10 = deconv(kernels[3], kernels[2], ratio="x2")
        self.inplanes = kernels[2]
        self.upResR1 = self._make_layer(block, kernels[2], layers[2], BNmode=BNmode)
        self.deconvR11 = deconv(kernels[2], kernels[1], ratio="x2")
        self.inplanes = kernels[1]
        self.upResR2 = self._make_layer(block, kernels[1], layers[1], BNmode=BNmode)
        self.deconvR12 = deconv(kernels[1], kernels[0], ratio="x2")
        self.inplanes = kernels[0]
        self.upResR3 = self._make_layer(block, kernels[0], layers[0], BNmode=BNmode)
        self.deconvR13 = deconv(kernels[0], kernels[0], ratio="x2")
        self.predR = nn.Conv2d(kernels[0], tar_ch[1], kernel_size=3, stride=1, padding=1, bias=False)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1, BNmode='IN'):
        downsample = None
        #         print("####2 BNmode => ", BNmode)
        if stride != 1 or self.inplanes != planes * block.expansion:
            bnl = nn.BatchNorm2d(planes * block.expansion)
            if BNmode == 'IN':
                bnl = nn.InstanceNorm2d(planes * block.expansion)
            if BNmode == 'GN':
                bnl = nn.GroupNorm(32, planes * block.expansion)
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                bnl,
            )

        layers = []
        if BNmode == 'GN':
            layers.append(block(self.inplanes, planes, stride, downsample, BNmode=BNmode))
        else:
            layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            if BNmode == 'GN':
                layers.append(block(self.inplanes, planes, BNmode=BNmode))
            else:
                layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    @staticmethod
    def _gaussian_filter(channel, kernel_size, sigma=1.5):
        gauss = torch.Tensor(
            [math.exp(-(x - kernel_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(kernel_size)])
        gauss_1D = (gauss / gauss.sum()).unsqueeze(1)
        gauss_2D = gauss_1D.mm(gauss_1D.t()).float().unsqueeze(0).unsqueeze(0)
        return gauss_2D.expand(channel, 1, kernel_size, kernel_size).contiguous()

    def _status_sync(self, x1, x2, kernel_size=3, sigma=1.5):
        nb, ch, row, col = x1.size()
        cosine = F.cosine_similarity(x1, x2, dim=1).unsqueeze(1)
        filters = self._gaussian_filter(1, kernel_size, sigma).to(cosine.device)
        # print("Cosine =>", cosine.size(), "Filter =>", filters.size())
        attenMap = F.conv2d(cosine, filters, padding=1)
        attenMap = attenMap.expand(nb, ch, row, col).contiguous()
        x1 = x1 * attenMap
        x2 = x2 * attenMap
        return x1, x2

    def forward(self, x):
        if self.src_ch == 1:
            x = torch.cat([x, x, x], dim=1)
        # forward network
        x1 = self.conv1(x)
        x1 = self.bn1(x1)
        x2 = self.relu(x1)
        # x2 = self.maxpool(x1)
        x3 = self.layer1(x2)
        x4 = self.layer2(x3)
        x5 = self.layer3(x4)
        x6 = self.layer4(x5)

        ## stage 1
        # left
        x7L = self.deconvL10(x6)
        # right
        x7R = self.deconvR10(x6)
        # sycn
        x7L, x7R = self._status_sync(x7L, x7R)

        ## stage 2
        # left
        x8L = x7L + x5  # add x7 and x5
        x9L = self.upResL1(x8L)
        x10L = self.deconvL11(x9L)
        # right
        x8R = x7R + x5  # add x7 and x5
        x9R = self.upResR1(x8R)
        x10R = self.deconvR11(x9R)
        # sycn
        x10L, x10R = self._status_sync(x10L, x10R)

        ## stage 3
        # left
        x11L = x10L + x4  # add x11 and x4
        x12L = self.upResL2(x11L)
        x13L = self.deconvL12(x12L)
        # right
        x11R = x10R + x4  # add x11 and x4
        x12R = self.upResR2(x11R)
        x13R = self.deconvR12(x12R)
        # sycn
        x13L, x13R = self._status_sync(x13L, x13R)

        ## stage 3
        # left
        x14L = x13L + x3  # add x13 and x3
        x15L = self.upResL3(x14L)
        x16L = self.deconvL13(x15L)
        # right
        x14R = x13R + x3  # add x13 and x3
        x15R = self.upResR3(x14R)
        x16R = self.deconvR13(x15R)
        # sycn
        x16L, x16R = self._status_sync(x16L, x16R)

        ## stage 4
        x_L = self.predL(x16L)
        x_R = self.predR(x16R)
        return x_L, x_R


class ResYNetSync_edge(nn.Module):

    def __init__(self, block, layers, src_ch, tar_ch, BNmode):
        super(ResYNetSync_edge, self).__init__()
        kernels = [64, 128, 256, 512]
        self.edge_detection = CannyFilter(use_cuda=True)

        self.src_ch = src_ch
        self.inplanes = kernels[0]
        self.conv1 = nn.Conv2d(3, kernels[0], kernel_size=7,
                               stride=2, padding=3,
                               bias=False)
        #         print("####1 BNmode => ", BNmode)
        if BNmode == 'BN':
            self.bn1 = nn.BatchNorm2d(kernels[0])
        elif BNmode == 'IN':
            self.bn1 = nn.InstanceNorm2d(kernels[0])
        elif BNmode == 'GN':
            self.bn1 = nn.GroupNorm(32, kernels[0])
        self.relu = nn.ReLU(inplace=True)
        # self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=0, ceil_mode=True)

        self.layer1 = self._make_layer(block, kernels[0], layers[0], stride=1, BNmode=BNmode)
        self.layer2 = self._make_layer(block, kernels[1], layers[1], stride=2, BNmode=BNmode)
        self.layer3 = self._make_layer(block, kernels[2], layers[2], stride=2, BNmode=BNmode)
        self.layer4 = self._make_layer(block, kernels[3], layers[3], stride=2, BNmode=BNmode)

        # Left arm
        self.deconvL10 = deconv(kernels[3], kernels[2], ratio="x2")
        self.inplanes = kernels[2]
        self.upResL1 = self._make_layer(block, kernels[2], layers[2], BNmode=BNmode)
        self.deconvL11 = deconv(kernels[2], kernels[1], ratio="x2")
        self.inplanes = kernels[1]
        self.upResL2 = self._make_layer(block, kernels[1], layers[1], BNmode=BNmode)
        self.deconvL12 = deconv(kernels[1], kernels[0], ratio="x2")
        self.inplanes = kernels[0]
        self.upResL3 = self._make_layer(block, kernels[0], layers[0], BNmode=BNmode)
        self.deconvL13 = deconv(kernels[0], kernels[0], ratio="x2")
        self.predL = nn.Conv2d(kernels[0], tar_ch[0], kernel_size=3, stride=1, padding=1, bias=False)

        # Right arm
        self.deconvR10 = deconv(kernels[3], kernels[2], ratio="x2")
        self.inplanes = kernels[2]
        self.upResR1 = self._make_layer(block, kernels[2], layers[2], BNmode=BNmode)
        self.deconvR11 = deconv(kernels[2], kernels[1], ratio="x2")
        self.inplanes = kernels[1]
        self.upResR2 = self._make_layer(block, kernels[1], layers[1], BNmode=BNmode)
        self.deconvR12 = deconv(kernels[1], kernels[0], ratio="x2")
        self.inplanes = kernels[0]
        self.upResR3 = self._make_layer(block, kernels[0], layers[0], BNmode=BNmode)
        self.deconvR13 = deconv(kernels[0], kernels[0], ratio="x2")
        self.predR = nn.Conv2d(kernels[0], tar_ch[1], kernel_size=3, stride=1, padding=1, bias=False)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1, BNmode='IN'):
        downsample = None
        #         print("####2 BNmode => ", BNmode)
        if stride != 1 or self.inplanes != planes * block.expansion:
            bnl = nn.BatchNorm2d(planes * block.expansion)
            if BNmode == 'IN':
                bnl = nn.InstanceNorm2d(planes * block.expansion)
            if BNmode == 'GN':
                bnl = nn.GroupNorm(32, planes * block.expansion)
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                bnl,
            )

        layers = []
        if BNmode == 'GN':
            layers.append(block(self.inplanes, planes, stride, downsample, BNmode=BNmode))
        else:
            layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            if BNmode == 'GN':
                layers.append(block(self.inplanes, planes, BNmode=BNmode))
            else:
                layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    @staticmethod
    def _gaussian_filter(channel, kernel_size, sigma=1.5):
        gauss = torch.Tensor(
            [math.exp(-(x - kernel_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(kernel_size)])
        gauss_1D = (gauss / gauss.sum()).unsqueeze(1)
        gauss_2D = gauss_1D.mm(gauss_1D.t()).float().unsqueeze(0).unsqueeze(0)
        return gauss_2D.expand(channel, 1, kernel_size, kernel_size).contiguous()

    def _status_sync(self, x1, x2, kernel_size=3, sigma=1.5):
        nb, ch, row, col = x1.size()
        cosine = F.cosine_similarity(x1, x2, dim=1).unsqueeze(1)
        filters = self._gaussian_filter(1, kernel_size, sigma).to(cosine.device)
        # print("Cosine =>", cosine.size(), "Filter =>", filters.size())
        attenMap = F.conv2d(cosine, filters, padding=1)
        attenMap = attenMap.expand(nb, ch, row, col).contiguous()
        x1 = x1 * attenMap
        x2 = x2 * attenMap
        return x1, x2

    def forward(self, x):
        if self.src_ch == 1:
            x = torch.cat([x, x, x], dim=1)

        # forward network
        _, _, _, _, _, thin_edges = self.edge_detection(x)
        thin_edges = torch.cat([thin_edges, thin_edges, thin_edges], dim=1)

        x1 = self.conv1(thin_edges)
        x1 = self.bn1(x1)
        x2 = self.relu(x1)
        # x2 = self.maxpool(x1)
        x3 = self.layer1(x2)
        x4 = self.layer2(x3)
        x5 = self.layer3(x4)
        x6 = self.layer4(x5)

        ## stage 1
        # left
        x7L = self.deconvL10(x6)
        # right
        x7R = self.deconvR10(x6)
        # sycn
        x7L, x7R = self._status_sync(x7L, x7R)

        ## stage 2
        # left
        x8L = x7L + x5  # add x7 and x5
        x9L = self.upResL1(x8L)
        x10L = self.deconvL11(x9L)
        # right
        x8R = x7R + x5  # add x7 and x5
        x9R = self.upResR1(x8R)
        x10R = self.deconvR11(x9R)
        # sycn
        x10L, x10R = self._status_sync(x10L, x10R)

        ## stage 3
        # left
        x11L = x10L + x4  # add x11 and x4
        x12L = self.upResL2(x11L)
        x13L = self.deconvL12(x12L)
        # right
        x11R = x10R + x4  # add x11 and x4
        x12R = self.upResR2(x11R)
        x13R = self.deconvR12(x12R)
        # sycn
        x13L, x13R = self._status_sync(x13L, x13R)

        ## stage 3
        # left
        x14L = x13L + x3  # add x13 and x3
        x15L = self.upResL3(x14L)
        x16L = self.deconvL13(x15L)
        # right
        x14R = x13R + x3  # add x13 and x3
        x15R = self.upResR3(x14R)
        x16R = self.deconvR13(x15R)
        # sycn
        x16L, x16R = self._status_sync(x16L, x16R)

        ## stage 4
        x_L = self.predL(x16L)
        x_R = self.predR(x16R)
        return x_L, x_R


class ResYNetSync_final(nn.Module):

    def __init__(self, block, layers, src_ch, tar_ch, BNmode):
        super(ResYNetSync_final, self).__init__()
        kernels = [64, 128, 256, 512]
        self.src_ch = src_ch
        self.inplanes = kernels[0]
        self.conv1 = nn.Conv2d(6, kernels[0], kernel_size=7,
                               stride=2, padding=3,
                               bias=False)
        #         print("####1 BNmode => ", BNmode)
        if BNmode == 'BN':
            self.bn1 = nn.BatchNorm2d(kernels[0])
        elif BNmode == 'IN':
            self.bn1 = nn.InstanceNorm2d(kernels[0])
        elif BNmode == 'GN':
            self.bn1 = nn.GroupNorm(32, kernels[0])
        self.relu = nn.ReLU(inplace=True)
        # self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=0, ceil_mode=True)

        self.layer1 = self._make_layer(block, kernels[0], layers[0], stride=1, BNmode=BNmode)
        self.layer2 = self._make_layer(block, kernels[1], layers[1], stride=2, BNmode=BNmode)
        self.layer3 = self._make_layer(block, kernels[2], layers[2], stride=2, BNmode=BNmode)
        self.layer4 = self._make_layer(block, kernels[3], layers[3], stride=2, BNmode=BNmode)

        # Left arm
        self.deconvL10 = deconv(kernels[3], kernels[2], ratio="x2")
        self.inplanes = kernels[2]
        self.upResL1 = self._make_layer(block, kernels[2], layers[2], BNmode=BNmode)
        self.deconvL11 = deconv(kernels[2], kernels[1], ratio="x2")
        self.inplanes = kernels[1]
        self.upResL2 = self._make_layer(block, kernels[1], layers[1], BNmode=BNmode)
        self.deconvL12 = deconv(kernels[1], kernels[0], ratio="x2")
        self.inplanes = kernels[0]
        self.upResL3 = self._make_layer(block, kernels[0], layers[0], BNmode=BNmode)
        self.deconvL13 = deconv(kernels[0], kernels[0], ratio="x2")
        self.predL = nn.Conv2d(kernels[0], tar_ch[0], kernel_size=3, stride=1, padding=1, bias=False)

        # Right arm
        self.deconvR10 = deconv(kernels[3], kernels[2], ratio="x2")
        self.inplanes = kernels[2]
        self.upResR1 = self._make_layer(block, kernels[2], layers[2], BNmode=BNmode)
        self.deconvR11 = deconv(kernels[2], kernels[1], ratio="x2")
        self.inplanes = kernels[1]
        self.upResR2 = self._make_layer(block, kernels[1], layers[1], BNmode=BNmode)
        self.deconvR12 = deconv(kernels[1], kernels[0], ratio="x2")
        self.inplanes = kernels[0]
        self.upResR3 = self._make_layer(block, kernels[0], layers[0], BNmode=BNmode)
        self.deconvR13 = deconv(kernels[0], kernels[0], ratio="x2")
        self.predR = nn.Conv2d(kernels[0], tar_ch[1], kernel_size=3, stride=1, padding=1, bias=False)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1, BNmode='IN'):
        downsample = None
        #         print("####2 BNmode => ", BNmode)
        if stride != 1 or self.inplanes != planes * block.expansion:
            bnl = nn.BatchNorm2d(planes * block.expansion)
            if BNmode == 'IN':
                bnl = nn.InstanceNorm2d(planes * block.expansion)
            if BNmode == 'GN':
                bnl = nn.GroupNorm(32, planes * block.expansion)
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                bnl,
            )

        layers = []
        if BNmode == 'GN':
            layers.append(block(self.inplanes, planes, stride, downsample, BNmode=BNmode))
        else:
            layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            if BNmode == 'GN':
                layers.append(block(self.inplanes, planes, BNmode=BNmode))
            else:
                layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    @staticmethod
    def _gaussian_filter(channel, kernel_size, sigma=1.5):
        gauss = torch.Tensor(
            [math.exp(-(x - kernel_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(kernel_size)])
        gauss_1D = (gauss / gauss.sum()).unsqueeze(1)
        gauss_2D = gauss_1D.mm(gauss_1D.t()).float().unsqueeze(0).unsqueeze(0)
        return gauss_2D.expand(channel, 1, kernel_size, kernel_size).contiguous()

    def _status_sync(self, x1, x2, kernel_size=3, sigma=1.5):
        nb, ch, row, col = x1.size()
        cosine = F.cosine_similarity(x1, x2, dim=1).unsqueeze(1)
        filters = self._gaussian_filter(1, kernel_size, sigma).to(cosine.device)
        # print("Cosine =>", cosine.size(), "Filter =>", filters.size())
        attenMap = F.conv2d(cosine, filters, padding=1)
        attenMap = attenMap.expand(nb, ch, row, col).contiguous()
        x1 = x1 * attenMap
        x2 = x2 * attenMap
        return x1, x2

    def forward(self, x):
        if self.src_ch == 1:
            x = torch.cat([x, x, x], dim=1)
        # forward network
        x1 = self.conv1(x)
        x1 = self.bn1(x1)
        x2 = self.relu(x1)
        # x2 = self.maxpool(x1)
        x3 = self.layer1(x2)
        x4 = self.layer2(x3)
        x5 = self.layer3(x4)
        x6 = self.layer4(x5)

        ## stage 1
        # left
        x7L = self.deconvL10(x6)
        # right
        x7R = self.deconvR10(x6)
        # sycn
        x7L, x7R = self._status_sync(x7L, x7R)

        ## stage 2
        # left
        x8L = x7L + x5  # add x7 and x5
        x9L = self.upResL1(x8L)
        x10L = self.deconvL11(x9L)
        # right
        x8R = x7R + x5  # add x7 and x5
        x9R = self.upResR1(x8R)
        x10R = self.deconvR11(x9R)
        # sycn
        x10L, x10R = self._status_sync(x10L, x10R)

        ## stage 3
        # left
        x11L = x10L + x4  # add x11 and x4
        x12L = self.upResL2(x11L)
        x13L = self.deconvL12(x12L)
        # right
        x11R = x10R + x4  # add x11 and x4
        x12R = self.upResR2(x11R)
        x13R = self.deconvR12(x12R)
        # sycn
        x13L, x13R = self._status_sync(x13L, x13R)

        ## stage 3
        # left
        x14L = x13L + x3  # add x13 and x3
        x15L = self.upResL3(x14L)
        x16L = self.deconvL13(x15L)
        # right
        x14R = x13R + x3  # add x13 and x3
        x15R = self.upResR3(x14R)
        x16R = self.deconvR13(x15R)
        # sycn
        x16L, x16R = self._status_sync(x16L, x16R)

        ## stage 4
        x_L = self.predL(x16L)
        x_R = self.predR(x16R)
        return x_L, x_R


class PatchDiscriminator(nn.Module):
    def __init__(self):
        super(PatchDiscriminator, self).__init__()

        act = nn.LeakyReLU(0.2, inplace=True)
        input_channel = 1 + 3
        n_df = 64
        norm = nn.InstanceNorm2d

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


def resnetVQ(src_ch, tar_ch, pretrained, BNmode, **kwargs):
    """

    :param pretrained: (bool): If True, returns a model pre-trained on ImageNet
    :param BNmode: (str) in [ BN, IN, GN ]
    """
    # model = ResNet_VQ2(block, [2, 2, 2, 2], src_ch, tar_ch, BNmode=BNmode)
    model = ResNet_VQ()
    if pretrained:
        from collections import OrderedDict
        pretrained_state = model_zoo.load_url(model_urls['resnet18'])
        model_state = model.state_dict()
        selected_state = OrderedDict()
        for k, v in pretrained_state.items():
            if k in model_state and v.size() == model_state[k].size():
                selected_state[k] = v
        model_state.update(selected_state)
        model.load_state_dict(model_state)
    return model


def resnetVQ2(src_ch, tar_ch, pretrained, BNmode, **kwargs):
    """

    :param pretrained: (bool): If True, returns a model pre-trained on ImageNet
    :param BNmode: (str) in [ BN, IN, GN ]
    """
    model = ResNet_VQ2(block, [2, 2, 2, 2], src_ch, tar_ch, BNmode=BNmode)
    # model = ResNet_VQ()
    if pretrained:
        from collections import OrderedDict
        pretrained_state = model_zoo.load_url(model_urls['resnet18'])
        model_state = model.state_dict()
        selected_state = OrderedDict()
        for k, v in pretrained_state.items():
            if k in model_state and v.size() == model_state[k].size():
                selected_state[k] = v
        model_state.update(selected_state)
        model.load_state_dict(model_state)
    return model


def resnetVQ3(src_ch, tar_ch, pretrained, BNmode, **kwargs):
    """

    :param pretrained: (bool): If True, returns a model pre-trained on ImageNet
    :param BNmode: (str) in [ BN, IN, GN ]
    """
    model = ResNet_VQ3(block, [2, 2, 2, 2], src_ch, tar_ch, BNmode=BNmode)
    if pretrained:
        from collections import OrderedDict
        pretrained_state = model_zoo.load_url(model_urls['resnet18'])
        model_state = model.state_dict()
        selected_state = OrderedDict()
        for k, v in pretrained_state.items():
            if k in model_state and v.size() == model_state[k].size():
                selected_state[k] = v
        model_state.update(selected_state)
        model.load_state_dict(model_state)
    return model


def resnetpix2pixHD(src_ch, tar_ch, pretrained, BNmode, **kwargs):
    """

    :param pretrained: (bool): If True, returns a model pre-trained on ImageNet
    :param BNmode: (str) in [ BN, IN, GN ]
    """
    model = ResNet_pix2pixHD(block, [2, 2, 2, 2], src_ch, tar_ch, BNmode=BNmode)
    if pretrained:
        from collections import OrderedDict
        pretrained_state = model_zoo.load_url(model_urls['resnet18'])
        model_state = model.state_dict()
        selected_state = OrderedDict()
        for k, v in pretrained_state.items():
            if k in model_state and v.size() == model_state[k].size():
                selected_state[k] = v
        model_state.update(selected_state)
        model.load_state_dict(model_state)
    return model


def resnetpix2pixHD2(src_ch, tar_ch, pretrained, BNmode, **kwargs):
    """

    :param pretrained: (bool): If True, returns a model pre-trained on ImageNet
    :param BNmode: (str) in [ BN, IN, GN ]
    """
    model = ResNet_VQ3(block, [2, 2, 2, 2], src_ch, tar_ch, BNmode=BNmode)
    if pretrained:
        from collections import OrderedDict
        pretrained_state = model_zoo.load_url(model_urls['resnet18'])
        model_state = model.state_dict()
        selected_state = OrderedDict()
        for k, v in pretrained_state.items():
            if k in model_state and v.size() == model_state[k].size():
                selected_state[k] = v
        model_state.update(selected_state)
        model.load_state_dict(model_state)
    return model


def resnetpix2pixHD3(src_ch, tar_ch, pretrained, BNmode, **kwargs):
    """

    :param pretrained: (bool): If True, returns a model pre-trained on ImageNet
    :param BNmode: (str) in [ BN, IN, GN ]
    """
    model = ResNet_pix2pixHD_Cannyedge(block, [2, 2, 2, 2], src_ch, tar_ch, BNmode=BNmode)
    if pretrained:
        from collections import OrderedDict
        pretrained_state = model_zoo.load_url(model_urls['resnet18'])
        model_state = model.state_dict()
        selected_state = OrderedDict()
        for k, v in pretrained_state.items():
            if k in model_state and v.size() == model_state[k].size():
                selected_state[k] = v
        model_state.update(selected_state)
        model.load_state_dict(model_state)
    return model


def resnetpix2pixHD4(src_ch, tar_ch, pretrained, BNmode, **kwargs):
    """

    :param pretrained: (bool): If True, returns a model pre-trained on ImageNet
    :param BNmode: (str) in [ BN, IN, GN ]
    """
    model = ResNet_pix2pixHD_Cannyedge(block, [2, 2, 2, 2], src_ch, tar_ch, BNmode=BNmode)
    if pretrained:
        from collections import OrderedDict
        pretrained_state = model_zoo.load_url(model_urls['resnet18'])
        model_state = model.state_dict()
        selected_state = OrderedDict()
        for k, v in pretrained_state.items():
            if k in model_state and v.size() == model_state[k].size():
                selected_state[k] = v
        model_state.update(selected_state)
        model.load_state_dict(model_state)
    return model


def resnetpix2pixWGAN(src_ch, tar_ch, pretrained, BNmode, **kwargs):
    """

    :param pretrained: (bool): If True, returns a model pre-trained on ImageNet
    :param BNmode: (str) in [ BN, IN, GN ]
    """
    model = ResNet_pix2pixHD_Cannyedge(block, [2, 2, 2, 2], src_ch, tar_ch, BNmode=BNmode)
    if pretrained:
        from collections import OrderedDict
        pretrained_state = model_zoo.load_url(model_urls['resnet18'])
        model_state = model.state_dict()
        selected_state = OrderedDict()
        for k, v in pretrained_state.items():
            if k in model_state and v.size() == model_state[k].size():
                selected_state[k] = v
        model_state.update(selected_state)
        model.load_state_dict(model_state)
    return model


def resnetWGAN(src_ch, tar_ch, pretrained, BNmode, **kwargs):
    """

    :param pretrained: (bool): If True, returns a model pre-trained on ImageNet
    :param BNmode: (str) in [ BN, IN, GN ]
    """
    model = ResNet_VQ3(block, [2, 2, 2, 2], src_ch, tar_ch, BNmode=BNmode)
    if pretrained:
        from collections import OrderedDict
        pretrained_state = model_zoo.load_url(model_urls['resnet18'])
        model_state = model.state_dict()
        selected_state = OrderedDict()
        for k, v in pretrained_state.items():
            if k in model_state and v.size() == model_state[k].size():
                selected_state[k] = v
        model_state.update(selected_state)
        model.load_state_dict(model_state)
    return model


def resnetpix2pixWGAN2(src_ch, tar_ch, pretrained, BNmode, **kwargs):
    """
    :param pretrained: (bool): If True, returns a model pre-trained on ImageNet
    :param BNmode: (str) in [ BN, IN, GN ]
    """
    model = ResNet_pix2pixHD_Cannyedge2(block, [2, 2, 2, 2], src_ch, tar_ch, BNmode=BNmode)
    if pretrained:
        from collections import OrderedDict
        pretrained_state = model_zoo.load_url(model_urls['resnet18'])
        model_state = model.state_dict()
        selected_state = OrderedDict()
        for k, v in pretrained_state.items():
            if k in model_state and v.size() == model_state[k].size():
                selected_state[k] = v
        model_state.update(selected_state)
        model.load_state_dict(model_state)
    return model


def resnetpix2pixWGAN3(src_ch, tar_ch, pretrained, BNmode, **kwargs):
    """
    :param pretrained: (bool): If True, returns a model pre-trained on ImageNet
    :param BNmode: (str) in [ BN, IN, GN ]
    """
    model = ResNet_pix2pixHD(block, [2, 2, 2, 2], src_ch, tar_ch, BNmode=BNmode)
    if pretrained:
        from collections import OrderedDict
        pretrained_state = model_zoo.load_url(model_urls['resnet18'])
        model_state = model.state_dict()
        selected_state = OrderedDict()
        for k, v in pretrained_state.items():
            if k in model_state and v.size() == model_state[k].size():
                selected_state[k] = v
        model_state.update(selected_state)
        model.load_state_dict(model_state)
    return model


def resnetpix2pixWGAN4(src_ch, tar_ch, pretrained, BNmode, **kwargs):
    """
    :param pretrained: (bool): If True, returns a model pre-trained on ImageNet
    :param BNmode: (str) in [ BN, IN, GN ]
    """
    model = ResNet_pix2pixHD(block, [2, 2, 2, 2], src_ch, tar_ch, BNmode=BNmode)
    if pretrained:
        from collections import OrderedDict
        pretrained_state = model_zoo.load_url(model_urls['resnet18'])
        model_state = model.state_dict()
        selected_state = OrderedDict()
        for k, v in pretrained_state.items():
            if k in model_state and v.size() == model_state[k].size():
                selected_state[k] = v
        model_state.update(selected_state)
        model.load_state_dict(model_state)
    return model


def resnetpix2pixcustom(src_ch, tar_ch, pretrained, BNmode, **kwargs):
    """
    :param pretrained: (bool): If True, returns a model pre-trained on ImageNet
    :param BNmode: (str) in [ BN, IN, GN ]
    """
    model = ResNet_pix2pixHD_Cannyedge_Custom(block, [2, 2, 2, 2], src_ch, tar_ch, BNmode=BNmode)
    model_edge = ResNet_pix2pixHD_Cannyedge_Custom_edge(block, [2, 2, 2, 2], src_ch, tar_ch, BNmode=BNmode)
    model_decoder = ResNet_pix2pixHD_Cannyedge_Decoder(block, [2, 2, 2, 2], src_ch, tar_ch, BNmode=BNmode)
    if pretrained:
        from collections import OrderedDict
        pretrained_state = model_zoo.load_url(model_urls['resnet18'])
        model_state = model.state_dict()
        model_state_edge = model_edge.state_dict()
        selected_state = OrderedDict()
        selected_state_edge = OrderedDict()
        for k, v in pretrained_state.items():
            if k in model_state and v.size() == model_state[k].size():
                selected_state[k] = v
            if k in model_state_edge and v.size() == model_state_edge[k].size():
                selected_state_edge[k] = v

        model_state.update(selected_state)
        model.load_state_dict(model_state)
        model_state_edge.update(selected_state_edge)
        model_edge.load_state_dict(model_state_edge)

    return model, model_edge, model_decoder


def resnetpix2pixcustom3(src_ch, tar_ch, pretrained, BNmode, **kwargs):
    """
    :param pretrained: (bool): If True, returns a model pre-trained on ImageNet
    :param BNmode: (str) in [ BN, IN, GN ]
    """
    model = ResNet_pix2pixHD_Cannyedge_Custom(block, [2, 2, 2, 2], src_ch, tar_ch, BNmode=BNmode)
    model_edge = ResNet_pix2pixHD_Cannyedge_Custom_edge(block, [2, 2, 2, 2], src_ch, tar_ch, BNmode=BNmode)
    model_decoder = ResNet_pix2pixHD_Cannyedge_Decoder(block, [2, 2, 2, 2], src_ch, tar_ch, BNmode=BNmode)
    if pretrained:
        from collections import OrderedDict
        pretrained_state = model_zoo.load_url(model_urls['resnet18'])
        model_state = model.state_dict()
        model_state_edge = model_edge.state_dict()
        selected_state = OrderedDict()
        selected_state_edge = OrderedDict()
        for k, v in pretrained_state.items():
            if k in model_state and v.size() == model_state[k].size():
                selected_state[k] = v
            if k in model_state_edge and v.size() == model_state_edge[k].size():
                selected_state_edge[k] = v

        model_state.update(selected_state)
        model.load_state_dict(model_state)
        model_state_edge.update(selected_state_edge)
        model_edge.load_state_dict(model_state_edge)

    return model, model_edge, model_decoder


def resnetpix2pixfinal(src_ch, tar_ch, pretrained, BNmode, **kwargs):
    """
    :param pretrained: (bool): If True, returns a model pre-trained on ImageNet
    :param BNmode: (str) in [ BN, IN, GN ]
    """
    model_input = ResNet_pix2pixHD_input(block, [2, 2, 2, 2], src_ch, tar_ch, BNmode=BNmode)
    model_edge = ResNet_pix2pixHD_edgeparts(block, [2, 2, 2, 2], src_ch, tar_ch, BNmode=BNmode)
    model_final = ResNet_pix2pixHD_final(block, [2, 2, 2, 2], src_ch, tar_ch, BNmode=BNmode)
    if pretrained:
        from collections import OrderedDict
        pretrained_state = model_zoo.load_url(model_urls['resnet18'])
        model_input_state = model_input.state_dict()
        model_edge_state = model_edge.state_dict()
        model_final_state = model_final.state_dict()
        selected_state = OrderedDict()
        selected_state_edge = OrderedDict()
        selected_state_final = OrderedDict()
        for k, v in pretrained_state.items():
            if k in model_input_state and v.size() == model_input_state[k].size():
                selected_state[k] = v
            if k in model_edge_state and v.size() == model_edge_state[k].size():
                selected_state_edge[k] = v
            if k in model_final_state and v.size() == model_final_state[k].size():
                selected_state_final[k] = v

        model_input_state.update(selected_state)
        model_input.load_state_dict(model_input_state)
        model_edge_state.update(selected_state_edge)
        model_edge.load_state_dict(model_edge_state)
        model_final_state.update(selected_state_final)
        model_final.load_state_dict(model_final_state)

    return model_input, model_edge, model_final


def resnetpix2pixfinal6(src_ch, tar_ch, pretrained, BNmode, **kwargs):
    """
    :param pretrained: (bool): If True, returns a model pre-trained on ImageNet
    :param BNmode: (str) in [ BN, IN, GN ]
    """
    model_input = ResNet_pix2pixHD_input(block, [2, 2, 2, 2], src_ch, tar_ch, BNmode=BNmode)
    model_edge = ResNet_pix2pixHD_edgeparts(block, [2, 2, 2, 2], src_ch, tar_ch, BNmode=BNmode)
    model_final = ResNet_pix2pixHD_final(block, [2, 2, 2, 2], src_ch, tar_ch, BNmode=BNmode)
    if pretrained:
        from collections import OrderedDict
        pretrained_state = model_zoo.load_url(model_urls['resnet18'])
        model_input_state = model_input.state_dict()
        model_edge_state = model_edge.state_dict()
        model_final_state = model_final.state_dict()
        selected_state = OrderedDict()
        selected_state_edge = OrderedDict()
        selected_state_final = OrderedDict()
        for k, v in pretrained_state.items():
            if k in model_input_state and v.size() == model_input_state[k].size():
                selected_state[k] = v
            if k in model_edge_state and v.size() == model_edge_state[k].size():
                selected_state_edge[k] = v
            if k in model_final_state and v.size() == model_final_state[k].size():
                selected_state_final[k] = v

        model_input_state.update(selected_state)
        model_input.load_state_dict(model_input_state)
        model_edge_state.update(selected_state_edge)
        model_edge.load_state_dict(model_edge_state)
        model_final_state.update(selected_state_final)
        model_final.load_state_dict(model_final_state)

    return model_input, model_edge, model_final


def resnetpix2pixbeta(src_ch, tar_ch, pretrained, BNmode, **kwargs):
    """
    :param pretrained: (bool): If True, returns a model pre-trained on ImageNet
    :param BNmode: (str) in [ BN, IN, GN ]
    """
    model_input = ResYNetSync_origin(block, [2, 2, 2, 2], src_ch, tar_ch, BNmode=BNmode)
    model_edge = ResYNetSync_edge(block, [2, 2, 2, 2], src_ch, tar_ch, BNmode=BNmode)
    model_final = ResYNetSync_final(block, [2, 2, 2, 2], src_ch, tar_ch, BNmode=BNmode)
    if pretrained:
        from collections import OrderedDict
        pretrained_state = model_zoo.load_url(model_urls['resnet18'])
        model_input_state = model_input.state_dict()
        model_edge_state = model_edge.state_dict()
        model_final_state = model_final.state_dict()
        selected_state = OrderedDict()
        selected_state_edge = OrderedDict()
        selected_state_final = OrderedDict()
        for k, v in pretrained_state.items():
            if k in model_input_state and v.size() == model_input_state[k].size():
                selected_state[k] = v
            if k in model_edge_state and v.size() == model_edge_state[k].size():
                selected_state_edge[k] = v
            if k in model_final_state and v.size() == model_final_state[k].size():
                selected_state_final[k] = v

        model_input_state.update(selected_state)
        model_input.load_state_dict(model_input_state)
        model_edge_state.update(selected_state_edge)
        model_edge.load_state_dict(model_edge_state)
        model_final_state.update(selected_state_final)
        model_final.load_state_dict(model_final_state)

    return model_input, model_edge, model_final


def resnetVAE3(src_ch, tar_ch, pretrained, BNmode, **kwargs):
    """

    :param pretrained: (bool): If True, returns a model pre-trained on ImageNet
    :param BNmode: (str) in [ BN, IN, GN ]
    """
    model = ResNet_AAE(block, [2, 2, 2, 2], src_ch, tar_ch, BNmode=BNmode)
    if pretrained:
        from collections import OrderedDict
        pretrained_state = model_zoo.load_url(model_urls['resnet18'])
        model_state = model.state_dict()
        selected_state = OrderedDict()
        for k, v in pretrained_state.items():
            if k in model_state and v.size() == model_state[k].size():
                selected_state[k] = v
        model_state.update(selected_state)
        model.load_state_dict(model_state)
    return model


def resnetVAE34(src_ch, tar_ch, pretrained, BNmode, **kwargs):
    """

    :param pretrained: (bool): If True, returns a model pre-trained on ImageNet
    :param BNmode: (str) in [ BN, IN, GN ]
    """
    model = Resnet_VAE(block, [3, 4, 6, 3], src_ch, tar_ch, BNmode=BNmode)
    if pretrained:
        from collections import OrderedDict
        pretrained_state = model_zoo.load_url(model_urls['resnet18'])
        model_state = model.state_dict()
        selected_state = OrderedDict()
        for k, v in pretrained_state.items():
            if k in model_state and v.size() == model_state[k].size():
                selected_state[k] = v
        model_state.update(selected_state)
        model.load_state_dict(model_state)
    return model


def res18netLab(src_ch, tar_ch, pretrained, BNmode, **kwargs):
    """

    :param pretrained: (bool): If True, returns a model pre-trained on ImageNet
    :param BNmode: (str) in [ BN, IN, GN ]
    """
    model = attentionResnetSync_Lab(block, Attention_block, [2, 2, 2, 2],
                                    src_ch, tar_ch, BNmode=BNmode)
    if pretrained:
        from collections import OrderedDict
        pretrained_state = model_zoo.load_url(model_urls['resnet18'])
        model_state = model.state_dict()
        selected_state = OrderedDict()
        for k, v in pretrained_state.items():
            if k in model_state and v.size() == model_state[k].size():
                selected_state[k] = v
        model_state.update(selected_state)
        model.load_state_dict(model_state)
    return model


def attentionres18net6(src_ch, tar_ch, pretrained, BNmode, **kwargs):
    """

    :param pretrained: (bool): If True, returns a model pre-trained on ImageNet
    :param BNmode: (str) in [ BN, IN, GN ]
    """
    model = attentionResnetSync(block, block, [2, 2, 2, 2],
                                src_ch, tar_ch, BNmode=BNmode)
    if pretrained:
        from collections import OrderedDict
        pretrained_state = model_zoo.load_url(model_urls['resnet18'])
        model_state = model.state_dict()
        selected_state = OrderedDict()
        for k, v in pretrained_state.items():
            if k in model_state and v.size() == model_state[k].size():
                selected_state[k] = v
        model_state.update(selected_state)
        model.load_state_dict(model_state)
    return model


def attentionres34net(src_ch, tar_ch, pretrained, BNmode, **kwargs):
    """

    :param pretrained: (bool): If True, returns a model pre-trained on ImageNet
    :param BNmode: (str) in [ BN, IN, GN ]
    """
    model = attentionResnetSync(block, Attention_block, [3, 4, 6, 3],
                                src_ch, tar_ch, BNmode=BNmode)
    if pretrained:
        from collections import OrderedDict
        pretrained_state = model_zoo.load_url(model_urls['resnet18'])
        model_state = model.state_dict()
        selected_state = OrderedDict()
        for k, v in pretrained_state.items():
            if k in model_state and v.size() == model_state[k].size():
                selected_state[k] = v
        model_state.update(selected_state)
        model.load_state_dict(model_state)
    return model


if __name__ == "__main__":
    # Hyper Parameters
    parser = argparse.ArgumentParser(description='ArgumentParser')
    parser.add_argument('-img_row', type=int, default=224,
                        help='img_row of input')
    parser.add_argument('-img_col', type=int, default=224,
                        help='img_col of input')
    parser.add_argument('-src_ch', type=int, default=1,
                        help='nb channel of source')
    parser.add_argument('-tar_ch_R', type=int, default=1,
                        help='nb channel of R target')
    parser.add_argument('-tar_ch_G', type=int, default=1,
                        help='nb channel of G target')
    parser.add_argument('-tar_ch_B', type=int, default=1,
                        help='nb channel of B target')
    parser.add_argument('-base_kernel', type=int, default=12,
                        help='batch_size for training ')
    parser.add_argument('-train_mode', type=str, default='train', choices=['train', 'test'])
    parser.add_argument('-lr', type=float, default=1e-4,
                        help='learning rate for discriminator')
    args = parser.parse_args()

    x = torch.FloatTensor(
        np.random.random((args.base_kernel, args.src_ch, args.img_row, args.img_col)))
    tar_ch = [args.tar_ch_R, args.tar_ch_G, args.tar_ch_B]
    for BNmode in ['BN', 'IN', 'GN']:
        generator = resnetWGAN(args.src_ch, tar_ch, True, BNmode)
        gen_y = generator(x)
        total_params = sum(p.numel() for p in generator.parameters())
        print("attentionres18net : BN=>{}".format(BNmode))
        print(" Network output : ", gen_y.shape)
        print(" Params: {:0.1f}M".format(total_params / (10 ** 6)))

        generator = resnetVAE34(args.src_ch, tar_ch, True, BNmode)
        gen_y = generator(x)
        total_params = sum(p.numel() for p in generator.parameters())
        print("attentionres34net : BN=>{}".format(BNmode))
        print(" Network output : ", gen_y.shape)
        print(" Params: {:0.1f}M".format(total_params / (10 ** 6)))
