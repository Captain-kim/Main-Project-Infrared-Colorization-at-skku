import argparse
import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
import cv2
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


class ResYNetSync(nn.Module):

    def __init__(self, block, layers, src_ch, tar_ch, BNmode):
        super(ResYNetSync, self).__init__()
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

        _, _, _, magnitude, _, _ = self.edge_detection(x)
        magnitude = torch.cat([magnitude, magnitude, magnitude], dim=1)

        x1 = self.conv1(magnitude)
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


class MS_CAM_L(nn.Module):

    def __init__(self, channels=2, r=2):
        super(MS_CAM_L, self).__init__()
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

    def forward(self, x):
        xl = self.local_att(x)
        xg = self.global_att(x)
        xlg = xl + xg
        wei = self.sigmoid(xlg)
        return x * wei


class MS_CAM_ab(nn.Module):

    def __init__(self, channels=4, r=2):
        super(MS_CAM_ab, self).__init__()
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

    def forward(self, x):
        xl = self.local_att(x)
        xg = self.global_att(x)
        xlg = xl + xg
        wei = self.sigmoid(xlg)
        return x * wei


class ResYNetSync_final(nn.Module):

    def __init__(self, block, layers, src_ch, tar_ch, BNmode):
        super(ResYNetSync_final, self).__init__()
        kernels = [64, 128, 256, 512]
        self.MS_CAM_L = MS_CAM_L()
        self.MS_CAM_ab = MS_CAM_ab()
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

    def forward(self, x, x_edge):
        if self.src_ch == 1:
            x = torch.cat([x, x, x], dim=1)
            x_edge = torch.cat([x_edge, x_edge, x_edge], dim=1)
        # forward network
        x_L = torch.cat([x[:, :1, :, :], x_edge[:, :1, :, :]], dim=1)
        x_ab = torch.cat([x[:, 1:, :, :], x_edge[:, 1:, :, :]], dim=1)
        x_MS_CAM_L = self.MS_CAM_L(x_L)
        x_MS_CAM_ab = self.MS_CAM_ab(x_ab)
        x = torch.cat([x_MS_CAM_L[:, :1, :, :], x_MS_CAM_ab[:, :2, :, :]], dim=1)
        x_edge = torch.cat([x_MS_CAM_L[:, 1:, :, :], x_MS_CAM_ab[:, 2:, :, :]], dim=1)
        x_input = torch.cat([x, x_edge], dim=1)
        x1 = self.conv1(x_input)
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


def resnetpix2pixbeta2(src_ch, tar_ch, pretrained, BNmode, **kwargs):
    """
    :param pretrained: (bool): If True, returns a model pre-trained on ImageNet
    :param BNmode: (str) in [ BN, IN, GN ]
    """
    model_input = ResYNetSync_origin(BasicBlock, [2, 2, 2, 2], src_ch, tar_ch, BNmode=BNmode)
    model_edge = ResYNetSync_edge(BasicBlock, [2, 2, 2, 2], src_ch, tar_ch, BNmode=BNmode)
    model_final = ResYNetSync_final(BasicBlock, [2, 2, 2, 2], src_ch, tar_ch, BNmode=BNmode)
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


def resnetpix2pixbeta7(src_ch, tar_ch, pretrained, BNmode, **kwargs):
    """
    :param pretrained: (bool): If True, returns a model pre-trained on ImageNet
    :param BNmode: (str) in [ BN, IN, GN ]
    """
    model_input = ResYNetSync_origin(BasicBlock, [2, 2, 2, 2], src_ch, tar_ch, BNmode=BNmode)
    model_edge = ResYNetSync_edge(BasicBlock, [2, 2, 2, 2], src_ch, tar_ch, BNmode=BNmode)
    model_final = ResYNetSync_final(BasicBlock, [2, 2, 2, 2], src_ch, tar_ch, BNmode=BNmode)
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


def resnetpix2pixdelta(src_ch, tar_ch, pretrained, BNmode, **kwargs):
    """
    :param pretrained: (bool): If True, returns a model pre-trained on ImageNet
    :param BNmode: (str) in [ BN, IN, GN ]
    """
    model_input = ResYNetSync_origin(BasicBlock, [2, 2, 2, 2], src_ch, tar_ch, BNmode="GN")
    model_edge = ResYNetSync_edge(BasicBlock, [2, 2, 2, 2], src_ch, tar_ch, BNmode="GN")
    model_final = ResYNetSync_final(BasicBlock, [2, 2, 2, 2], src_ch, tar_ch, BNmode="IN")
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


def res18ynetsync(src_ch, tar_ch, pretrained, BNmode, **kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        BNmode (str) in [ BN, IN, GN ]
    """
    model = ResYNetSync(BasicBlock, [2, 2, 2, 2],
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


def res34ynetsync(src_ch, tar_ch, pretrained, BNmode, **kwargs):
    """Constructs a ResNet-34 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        BNmode (str) in [ BN, IN, GN ]
    """
    model = ResYNetSync(BasicBlock, [3, 4, 6, 3],
                        src_ch, tar_ch, BNmode=BNmode)
    if pretrained:
        from collections import OrderedDict
        pretrained_state = model_zoo.load_url(model_urls['resnet34'])
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
                        help='img_col of input ')
    parser.add_argument('-src_ch', type=int, default=1,
                        help='nb channel of source')
    parser.add_argument('-tar1_ch', type=int, default=1,
                        help='nb channel of target 1')
    parser.add_argument('-tar2_ch', type=int, default=2,
                        help='nb channel of target 2')
    parser.add_argument('-base_kernel', type=int, default=12,
                        help='batch_size for training ')
    parser.add_argument('-lr', type=float, default=1e-4,
                        help='learning rate for discriminator')
    args = parser.parse_args()

    x = torch.FloatTensor(
        np.random.random((args.base_kernel, args.src_ch, args.img_row, args.img_col)))

    for BNmode in ['BN', 'IN', 'GN']:
        generator = res18ynetsync(args.src_ch, [args.tar1_ch, args.tar2_ch], True, BNmode)
        gen_L, gen_R = generator(x)
        total_params = sum(p.numel() for p in generator.parameters())
        print("res18ynetsync : BN=>{}".format(BNmode))
        print(" Network L-output : ", gen_L.shape)
        print(" Network R-output : ", gen_R.shape)
        print(" Params: {:0.1f}M".format(total_params / (10 ** 6)))

        generator = res34ynetsync(args.src_ch, [args.tar1_ch, args.tar2_ch], True, BNmode)
        gen_L, gen_R = generator(x)
        total_params = sum(p.numel() for p in generator.parameters())
        print("res34ynetsync : BN=>{}".format(BNmode))
        print(" Network L-output : ", gen_L.shape)
        print(" Network R-output : ", gen_R.shape)
        print(" Params: {:0.1f}M".format(total_params / (10 ** 6)))

        generator = resnetpix2pixbeta7(args.src_ch, [args.tar1_ch, args.tar2_ch], True, BNmode)
        gen_L, gen_R = generator(x)
        total_params = sum(p.numel() for p in generator.parameters())
        print("resnetpix2pixbeta7 : BN=>{}".format(BNmode))
        print(" Network L-output : ", gen_L.shape)
        print(" Network R-output : ", gen_R.shape)
        print(" Params: {:0.1f}M".format(total_params / (10 ** 6)))

