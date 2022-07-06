import argparse
import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from .block import *


__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152']


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}

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
                 base_width = 64, dilation=1, BNmode='BN', fuse_type = 'AFF'):
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
        x5 = self.layer3(x4) # Attention
        x6 = self.layer4(x5) # Attention

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

        return torch.cat((x_final_R, x_final_G, x_final_B), dim=1)

def attentionres18ynet(src_ch, tar_ch, pretrained, BNmode, **kwargs):
    """

    :param pretrained: (bool): If True, returns a model pre-trained on ImageNet
    :param BNmode: (str) in [ BN, IN, GN ]
    """
    model = attentionResnetSync(block, Attention_block, [2, 2, 2, 2],
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


def attentionres34ynet(src_ch, tar_ch, pretrained, BNmode, **kwargs):
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
    parser.add_argument('-lr', type=float, default=1e-4,
                        help='learning rate for discriminator')
    args = parser.parse_args()

    x = torch.FloatTensor(
        np.random.random((args.base_kernel, args.src_ch, args.img_row, args.img_col)))
    tar_ch = [args.tar_ch_R, args.tar_ch_G, args.tar_ch_B]
    for BNmode in ['BN', 'IN', 'GN']:
        generator = attentionres18net(args.src_ch, tar_ch,
                                      True, BNmode)
        gen_y = generator(x)
        total_params = sum(p.numel() for p in generator.parameters())
        print("attentionres18net : BN=>{}".format(BNmode))
        print(" Network output : ", gen_y.shape)
        print(" Params: {:0.1f}M".format(total_params / (10**6)))

        generator = attentionres34net(args.src_ch, [args.tar_ch_R, args.tar_ch_G, args.tar_ch_B],
                                      True, BNmode)
        gen_y = generator(x)
        total_params = sum(p.numel() for p in generator.parameters())
        print("attentionres34net : BN=>{}".format(BNmode))
        print(" Network output : ", gen_y.shape)
        print(" Params: {:0.1f}M".format(total_params / (10 ** 6)))