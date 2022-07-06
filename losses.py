#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
  @Email:  guangmingwu2010@gmail.com
  @Copyright: go-hiroaki
  @License: MIT
"""
import numpy as np
import torch
from torch import nn
from torchvision import models
import torch.nn.functional as F
import metrics

def compute_gradient_penalty(D, input_samples, real_samples, fake_samples):
    # Random weight term for interpolation between real and fake samples
    Tensor = torch.cuda.FloatTensor
    alpha = Tensor(np.random.random((real_samples.size(0), 1, 1, 1)))
    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates = D(torch.cat((input_samples, interpolates), dim=1))
    # For PatchGAN
    fake = Variable(Tensor(real_samples.shape[0], 1, 30, 30).fill_(1.0), requires_grad = False)
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

eps = 1e-6

def get_grid(input, is_real=True):
    if is_real:
        grid = torch.FloatTensor(input.shape).fill_(1.0)

    elif not is_real:
        grid = torch.FloatTensor(input.shape).fill_(0.0)

    return grid

class pix2pixHDLoss(nn.Module):
    def __init__(self):
        super(pix2pixHDLoss, self).__init__()
        self.device = torch.device('cuda:0')
        # self.dtype = torch.float16

        self.criterion = nn.MSELoss()
        self.n_D = 2

    def __call__(self, D, G, input, target):
        loss_D = 0

        # fake = G(input)
        fake_L, fake_a, fake_b, _, _ = G(input)
        fake = torch.cat((fake_L, fake_a, fake_b), dim=1)

        real_features = D(torch.cat((input, target), dim=1))
        fake_features = D(torch.cat((input, fake.detach()), dim=1))

        for i in range(self.n_D):
            real_grid = get_grid(real_features[i][-1], is_real=True).to(self.device)
            fake_grid = get_grid(fake_features[i][-1], is_real=False).to(self.device)

            loss_D += (self.criterion(real_features[i][-1], real_grid) +
                       self.criterion(fake_features[i][-1], fake_grid)) * 0.5

        return loss_D, target, fake

class pix2pixHDLoss_custom(nn.Module):
    def __init__(self):
        super(pix2pixHDLoss_custom, self).__init__()
        self.device = torch.device('cuda:0')
        # self.dtype = torch.float16

        self.criterion = nn.MSELoss()
        self.n_D = 2

    def __call__(self, D, G, G_edge, G_Decoder, input, target):
        loss_D = 0

        # fake = G(input)
        fake_L_G, fake_a_G, fake_b_G = G(input)
        fake_L_e, fake_a_e, fake_b_e = G_edge(input)
        fake_G = torch.cat([fake_L_G, fake_a_G, fake_b_G], dim=1)
        fake_edge = torch.cat([fake_L_e, fake_a_e, fake_b_e], dim=1)
        fake_input = torch.cat([fake_G.detach(), fake_edge.detach()], dim=1)
        fake_L, fake_a, fake_b = G_Decoder(fake_input)

        fake = torch.cat([fake_L, fake_a, fake_b], dim=1)

        real_features = D(torch.cat((input, target), dim=1))
        fake_features = D(torch.cat((input, fake.detach()), dim=1))

        for i in range(self.n_D):
            real_grid = get_grid(real_features[i][-1], is_real=True).to(self.device)
            fake_grid = get_grid(fake_features[i][-1], is_real=False).to(self.device)

            loss_D += (self.criterion(real_features[i][-1], real_grid) +
                       self.criterion(fake_features[i][-1], fake_grid)) * 0.5

        return loss_D, target, fake

class pix2pixHDLoss_3gen(nn.Module):
    def __init__(self):
        super(pix2pixHDLoss_3gen, self).__init__()
        self.device = torch.device('cuda:0')
        # self.dtype = torch.float16

        self.criterion = nn.MSELoss()
        self.n_D = 2

    def __call__(self, D, G, G_edge, G_final, input, target):
        loss_D = 0

        # fake = G(input)
        fake_L_i, fake_ab_i = G(input)
        fake_L_e, fake_ab_e = G_edge(input)
        fake_i = torch.cat([fake_L_i, fake_ab_i], dim=1)
        fake_e = torch.cat([fake_L_e, fake_ab_e], dim=1)
        fake_input = torch.cat([fake_i.detach(), fake_e.detach()], dim=1)
        fake_L, fake_ab = G_final(fake_input)

        fake = torch.cat([fake_L, fake_ab], dim=1)

        real_features = D(torch.cat((input, target), dim=1))
        fake_features = D(torch.cat((input, fake.detach()), dim=1))

        for i in range(self.n_D):
            real_grid = get_grid(real_features[i][-1], is_real=True).to(self.device)
            fake_grid = get_grid(fake_features[i][-1], is_real=False).to(self.device)

            loss_D += (self.criterion(real_features[i][-1], real_grid) +
                       self.criterion(fake_features[i][-1], fake_grid)) * 0.5

        return loss_D, target, fake

class pix2pixHDLoss_3gen_FM(nn.Module):
    def __init__(self):
        super(pix2pixHDLoss_3gen_FM, self).__init__()
        self.device = torch.device('cuda:0')
        # self.dtype = torch.float16

        self.criterion = nn.MSELoss()
        self.FMcriterion = nn.L1Loss()
        self.n_D = 2

    def __call__(self, D, G, G_edge, G_final, input, target):
        loss_D = 0
        loss_G = 0
        loss_G_FM = 0

        # fake = G(input)
        fake_L_i, fake_ab_i = G(input)
        fake_L_e, fake_ab_e = G_edge(input)
        fake_i = torch.cat([fake_L_i, fake_ab_i], dim=1)
        fake_e = torch.cat([fake_L_e, fake_ab_e], dim=1)
        fake_input = torch.cat([fake_i.detach(), fake_e.detach()], dim=1)
        fake_L, fake_ab = G_final(fake_input)

        fake = torch.cat([fake_L, fake_ab], dim=1)

        real_features = D(torch.cat((input, target), dim=1))
        fake_features = D(torch.cat((input, fake.detach()), dim=1))

        for i in range(self.n_D):
            real_grid = get_grid(real_features[i][-1], is_real=True).to(self.device)
            fake_grid = get_grid(fake_features[i][-1], is_real=False).to(self.device)

            loss_D += (self.criterion(real_features[i][-1], real_grid) +
                       self.criterion(fake_features[i][-1], fake_grid)) * 0.5

        fake_features = D(torch.cat((input, fake), dim=1))

        for i in range(self.n_D):
            real_grid = get_grid(fake_features[i][-1]).to(self.device)
            loss_G += self.criterion(fake_features[i][-1], real_grid)

            for j in range(len(fake_features[0])):
                loss_G_FM += self.FMcriterion(fake_features[i][j], real_features[i][j].detach())

            loss_G += loss_G_FM * (1.0 / 2) * 10

        return loss_D, loss_G, target, fake

class pix2pixHDWGANLoss_3gen_FM(nn.Module):
    def __init__(self):
        super(pix2pixHDWGANLoss_3gen_FM, self).__init__()
        self.device = torch.device('cuda:0')
        # self.dtype = torch.float16

        self.criterion = nn.MSELoss()
        self.FMcriterion = nn.L1Loss()
        self.n_D = 2

    def __call__(self, D, G, G_edge, G_final, input, target):
        loss_D = 0
        loss_G = 0
        loss_G_FM = 0

        # fake = G(input)
        fake_L_i, fake_ab_i = G(input)
        fake_L_e, fake_ab_e = G_edge(input)
        fake_i = torch.cat([fake_L_i, fake_ab_i], dim=1)
        fake_e = torch.cat([fake_L_e, fake_ab_e], dim=1)
        fake_input = torch.cat([fake_i.detach(), fake_e.detach()], dim=1)
        fake_L, fake_ab = G_final(fake_input)

        fake = torch.cat([fake_L, fake_ab], dim=1)

        real_features = D(torch.cat((input, target), dim=1))
        fake_features = D(torch.cat((input, fake.detach()), dim=1))

        for i in range(self.n_D):
            loss_D += - torch.mean(real_features[i][-1]) + torch.mean(fake_features[i][-1])

        fake_features = D(torch.cat((input, fake), dim=1))

        for i in range(self.n_D):
            real_grid = get_grid(fake_features[i][-1]).to(self.device)
            loss_G += self.criterion(fake_features[i][-1], real_grid)

            for j in range(len(fake_features[0])):
                loss_G_FM += self.FMcriterion(fake_features[i][j], real_features[i][j].detach())

            loss_G += loss_G_FM * (1.0 / 2) * 10

        return loss_D, loss_G, target, fake

class pix2pixHDWGANLoss_3gen(nn.Module):
    def __init__(self):
        super(pix2pixHDWGANLoss_3gen, self).__init__()
        self.device = torch.device('cuda:0')
        # self.dtype = torch.float16

        self.criterion = nn.MSELoss()
        self.FMcriterion = nn.L1Loss()
        self.n_D = 2

    def __call__(self, D, G, G_edge, G_final, input, target):
        loss_D = 0
        loss_G = 0
        loss_G_FM = 0

        # fake = G(input)
        fake_L_i, fake_ab_i = G(input)
        fake_L_e, fake_ab_e = G_edge(input)
        fake_i = torch.cat([fake_L_i, fake_ab_i], dim=1)
        fake_e = torch.cat([fake_L_e, fake_ab_e], dim=1)
        fake_input = torch.cat([fake_i.detach(), fake_e.detach()], dim=1)
        fake_L, fake_ab = G_final(fake_input)

        fake = torch.cat([fake_L, fake_ab], dim=1)

        real_features = D(torch.cat((input, target), dim=1))
        fake_features = D(torch.cat((input, fake.detach()), dim=1))

        for i in range(self.n_D):
            loss_D += - torch.mean(real_features[i][-1]) + torch.mean(fake_features[i][-1])

        return loss_D, target, fake

class pix2pixHDWGANLoss_custom(nn.Module):
    def __init__(self):
        super(pix2pixHDWGANLoss_custom, self).__init__()
        self.device = torch.device('cuda:0')
        # self.dtype = torch.float16

        self.criterion = nn.MSELoss()
        self.n_D = 2

    def __call__(self, D, G, G_edge, G_Decoder, input, target):
        loss_D = 0

        # fake = G(input)
        _, _, _, x10_L_G, x10_a_G, x10_b_G = G(input)
        _, _, _, x10_L_edge, x10_a_edge, x10_b_edge = G_edge(input)
        fake_G = torch.cat([x10_L_G.detach(), x10_a_G.detach(), x10_b_G.detach()], dim=1)
        fake_edge = torch.cat([x10_L_edge.detach(), x10_a_edge.detach(), x10_b_edge.detach()], dim=1)
        fake_input = torch.cat([fake_G, fake_edge], dim=1)
        fake_L, fake_a, fake_b = G_Decoder(fake_input)

        fake = torch.cat((fake_L, fake_a, fake_b), dim=1)

        real_features = D(torch.cat((input, target), dim=1))
        fake_features = D(torch.cat((input, fake.detach()), dim=1))

        for i in range(self.n_D):
            loss_D += - torch.mean(real_features[i][-1]) + torch.mean(fake_features[i][-1])

        return loss_D, target, fake

class pix2pixHDWGANGPLoss(nn.Module):
    def __init__(self):
        super(pix2pixHDWGANGPLoss, self).__init__()
        self.device = torch.device('cuda:0')
        # self.dtype = torch.float16

        self.criterion = nn.MSELoss()
        self.FMcriterion = nn.L1Loss()
        self.n_D = 2

    def __call__(self, D, G, input, target):
        loss_D = 0
        loss_G = 0
        loss_G_FM = 0

        # fake = G(input)
        fake_L, fake_a, fake_b, _, _ = G(input)
        fake = torch.cat((fake_L, fake_a, fake_b), dim=1)

        real_features = D(torch.cat((input, target), dim=1))
        fake_features = D(torch.cat((input, fake.detach()), dim=1))
        gradient_penalty = compute_gradient_penalty(D, input, target, fake)
        for i in range(self.n_D):
            # gradient_penalty = compute_gradient_penalty(D, input.data, target.data, fake.data)
            loss_D += - torch.mean(real_features[i][-1]) + torch.mean(fake_features[i][-1])

        loss_D += 10 * gradient_penalty
        fake_features = D(torch.cat((input, fake), dim=1))

        for i in range(self.n_D):
            loss_G += - torch.mean(fake_features[i][-1])
            for j in range(len(fake_features[0])):
                loss_G_FM += self.FMcriterion(fake_features[i][j], real_features[i][j].detach())

            loss_G += loss_G_FM * (1.0 / 2) * 10

        return loss_D, loss_G, target, fake

class L1Loss(nn.Module):
    def __init__(self):
        super(L1Loss, self).__init__()
        self.criterion = nn.L1Loss(size_average=True)

    def __repr__(self):
        return "L1"

    def forward(self, output, target):
        loss = self.criterion(output, target)
        return loss

class L1Loss3D(nn.Module):
    def __init__(self):
        super(L1Loss3D, self).__init__()
        self.criterion = nn.L1Loss(size_average=True)

    def __repr__(self):
        return "L13D"

    def forward(self, output, target):
        nb, ch, frame, row, col = output.shape
        loss = []
        for f in range(frame):
            loss.append(self.criterion(output[:,:,f,:,:], target[:,:,f,:,:]))
        return sum(loss) / len(loss)


class MSELoss(nn.Module):
    def __init__(self):
        super(MSELoss, self).__init__()
        self.criterion = nn.MSELoss(size_average=True)
    
    def __repr__(self):
        return "MSE"

    def forward(self, output, target):
        loss = self.criterion(output, target)
        return loss


class PSNRLoss(nn.Module):
    def __init__(self):
        super(PSNRLoss, self).__init__()
        self.criterion = nn.MSELoss(size_average=True)

    def __repr__(self):
        return "PSNR"

    def forward(self, output, target):
        mse = self.criterion(output, target)
        loss = 10 * torch.log10(1.0 / mse)
        return loss

class VAELoss(nn.Module):
    def __init__(self):
        super(VAELoss, self).__init__()

    def __repr__(self):
        return "VAE"

    def forward(self, recon_x, x, mu, logvar):
        #MSE = F.mse_loss(recon_x, x, reduction='none')
        #BCE = torch.div(MSE, torch.numel(x))
        #BCE = torch.sum(BCE)
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        KLD = torch.div(KLD, x.shape[0])
        #return BCE + KLD
        return KLD

class CELoss(nn.Module):
    def __init__(self):
        super(CELoss, self).__init__()
        self.criterionBinary = nn.BCELoss(size_average=True)
        self.criterionMulti = nn.NLLLoss(size_average=True)

    def __repr__(self):
        return "CE"

    def forward(self, output, target):
        if target.shape[1] == 1:
            # binary cross enthropy
            loss = self.criterionBinary(output, target)
        else:
            # multi-class cross enthropy
            target = torch.argmax(target, dim=1).long()
            loss = self.criterionMulti(torch.log(output), target)
        return loss


class DSSIMLoss(nn.Module):
    def __init__(self):
        super(DSSIMLoss, self).__init__()
        self.criterion = metrics.SSIM()

    def __repr__(self):
        return "DSSIM"

    def forward(self, output, target):
        loss = (1. - self.criterion(output, target)) / 2.
        return loss


class DSSIMLoss3D(nn.Module):
    def __init__(self):
        super(DSSIMLoss3D, self).__init__()
        self.criterion = metrics.SSIM()

    def __repr__(self):
        return "DSSIM3D"

    def forward(self, output, target):
        nb, ch, frame, row, col = output.shape
        loss = []
        for f in range(frame):
            loss.append((1. - self.criterion(output[:,:,f,:,:], target[:,:,f,:,:])) / 2.)
        return sum(loss) / len(loss)


class NearestSelector(object):
    def __init__(self, shift=2, stride=1, criter='l1'):
        self.shift = shift
        self.stride = stride
        self.criter = criter

    def __repr__(self):
        return "NS"

    @staticmethod
    def unravel_index(tensor, cols):
        """
        args:
            tensor : 2D tensor, [nb, rows*cols]
            cols : int
        return 2D tensor nb * [rowIndex, colIndex]
        """
        index = torch.argmin(tensor, dim=1).view(-1,1)
        rIndex = index / cols
        cIndex = index % cols
        minRC = torch.cat([rIndex, cIndex], dim=1)
        # print("minRC", minRC.shape, minRC)
        return minRC

    def shift_diff(self, output, target, crop_row, crop_col):
        diff = []
        for i in range(0, 2 * self.shift):
            for j in range(0, 2 * self.shift):
                output_crop = output[:, :, 
                                     self.shift * self.stride: self.shift * self.stride + crop_row,
                                     self.shift * self.stride: self.shift * self.stride + crop_col,]
                target_crop = target[:, :, 
                                     i * self.stride: i * self.stride + crop_row,
                                     j * self.stride: j * self.stride + crop_col,]
                diff_ij = torch.sum(abs(target_crop-output_crop), dim=[1,2,3]).view(-1,1)
                diff.append(diff_ij)
        return torch.cat(diff, dim=1)
        
    def crop(self, output, target):
        nb, ch, row, col = output.shape
        crop_row = row - 2 * self.shift * self.stride
        crop_col = col - 2 * self.shift * self.stride
        diff = self.shift_diff(output.detach(), target.detach(), crop_row, crop_col)
        minRC = self.unravel_index(diff, 2 * self.shift)
        crop = [self.shift * self.stride, self.shift * self.stride + crop_row,
                self.shift * self.stride, self.shift * self.stride + crop_col]
        output_ = output[:,
                         :,
                        crop[0] : crop[1],
                        crop[2] : crop[3]]
        target_ = torch.zeros(*output_.shape).to(target.device)
        for idx, (minR, minC) in enumerate(minRC):
            target_[idx] = target[idx,
                                  :,
                                  minR * self.stride: minR * self.stride + crop_row,
                                  minC * self.stride: minC * self.stride + crop_row]
        return output_, target_

    
class ConLoss(nn.Module):
    """
    Consistency of samples within batch
    """

    def __init__(self):
        super(ConLoss, self).__init__()
        self.criterMSE = nn.MSELoss(size_average=True)

    def __repr__(self):
        return 'ConLoss'

    def forward(self, feats):
        feat_max, _ = torch.max(feats, dim=0)
        feat_min, _ = torch.min(feats, dim=0)
        zeros = torch.zeros(feat_max.shape).to(feats.device)
        return self.criterMSE(torch.abs(feat_max - feat_min), zeros)


class CrossLoss(nn.Module):
    """
    Cross comparison between samples within batch
    """

    def __init__(self):
        super(CrossLoss, self).__init__()
        self.criterion = nn.L1Loss(size_average=True)

    def __repr__(self):
        return 'CrossLoss'

    def forward(self, output, target):
        nb, ch, row, col = output.shape
        output = output[:nb-1, :, :, :]
        target = target[1:nb, :, :, :]
        return self.criterion(output, target)


class FLoss(nn.Module):
    """
    Focal Loss
    Lin, Tsung-Yi, et al. \
    "Focal loss for dense object detection." \
    Proceedings of the IEEE international conference on computer vision. 2017.
    (modified from https://github.com/umbertogriffo/focal-loss-keras/blob/master/losses.py)
    """
    def __init__(self, gamma=2., weight=None, size_average=True):
        super(FLoss, self).__init__()
        self.gamma = gamma
        self.weight = weight
        self.size_average = size_average

    def __repr__(self):
        return 'Focal'

    def _get_weights(self, y_true, nb_ch):
        """
        args:
            y_true : 3-d ndarray in [batch_size, img_rows, img_cols]
            nb_ch : int 
        return [float] weights
        """
        batch_size, img_rows, img_cols = y_true.shape
        pixels = batch_size * img_rows * img_cols
        weights = [torch.sum(y_true==ch).item() / pixels for ch in range(nb_ch)]
        return weights

    def forward(self, output, target):
        output = torch.clamp(output, min=eps, max=(1. - eps))
        if target.shape[1] == 1:
            # binary focal loss
            # weights = self._get_weigthts(target[:,0,:,:], 2)
            alpha = 0.1
            loss = - (1.-alpha) * ((1.-output)**self.gamma)*(target*torch.log(output)) \
              - alpha * (output**self.gamma)*((1.-target)*torch.log(1.-output))
        else:
            # multi-class focal loss
            # weights = self._get_weigthts(torch.argmax(target, dim=1), target.shape[1])
            loss = - ((1.-output)**self.gamma)*(target*torch.log(output))

        if self.size_average: 
            return loss.mean()
        else: 
            return loss.sum()
        
        
class VGG16Loss(nn.Module):
    def __init__(self, requires_grad=False, cuda=True):
        super(VGG16Loss, self).__init__()
        self.criterion = nn.L1Loss(size_average=True)
        vgg_pretrained_features = models.vgg16(pretrained=True).features
        self.slice1 = nn.Sequential()
        self.slice2 = nn.Sequential()
        self.slice3 = nn.Sequential()
        self.slice4 = nn.Sequential()
        
        for x in range(4):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(4, 9):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(9, 16):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(16, 23):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])

        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

        if cuda:
            self.slice1.cuda()
            self.slice2.cuda()
            self.slice3.cuda()
            self.slice4.cuda()

    def __repr__(self):
        return "VGG16"

    def forward(self, output, target):
        nb, ch, row, col = output.shape
        if ch == 1:
            output = torch.cat([output, output, output], dim=1)
            target = torch.cat([target, target, target], dim=1)
        ho = self.slice1(output)
        ht = self.slice1(target)
        h_relu1_2_loss = self.criterion(ho,ht)
        ho = self.slice2(ho)
        ht = self.slice2(ht)
        h_relu2_2_loss = self.criterion(ho,ht)
        ho = self.slice3(ho)
        ht = self.slice3(ht)
        h_relu3_3_loss = self.criterion(ho,ht)
        ho = self.slice4(ho)
        ht = self.slice4(ht)
        h_relu4_3_loss = self.criterion(ho,ht)
        return sum([h_relu1_2_loss, h_relu2_2_loss, h_relu3_3_loss, h_relu4_3_loss]) / 4


class VGG16Loss3D(nn.Module):
    def __init__(self, requires_grad=False, cuda=True):
        super(VGG16Loss3D, self).__init__()
        self.criterion = nn.L1Loss(size_average=True)
        vgg_pretrained_features = models.vgg16(pretrained=True).features
        self.slice1 = nn.Sequential()
        self.slice2 = nn.Sequential()
        self.slice3 = nn.Sequential()
        self.slice4 = nn.Sequential()
        
        for x in range(4):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(4, 9):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(9, 16):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(16, 23):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])

        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

        if cuda:
            self.slice1.cuda()
            self.slice2.cuda()
            self.slice3.cuda()
            self.slice4.cuda()

    def __repr__(self):
        return "VGG163D"

    def forward2d(self, output, target):
        nb, ch, row, col = output.shape
        if ch == 1:
            output = torch.cat([output, output, output], dim=1)
            target = torch.cat([target, target, target], dim=1)
        ho = self.slice1(output)
        ht = self.slice1(target)
        h_relu1_2_loss = self.criterion(ho,ht)
        ho = self.slice2(ho)
        ht = self.slice2(ht)
        h_relu2_2_loss = self.criterion(ho,ht)
        ho = self.slice3(ho)
        ht = self.slice3(ht)
        h_relu3_3_loss = self.criterion(ho,ht)
        ho = self.slice4(ho)
        ht = self.slice4(ht)
        h_relu4_3_loss = self.criterion(ho,ht)
        return sum([h_relu1_2_loss, h_relu2_2_loss, h_relu3_3_loss, h_relu4_3_loss]) / 4

    def forward(self, output, target):
        nb, ch, frame, row, col = output.shape
        loss = []
        for f in range(frame):
            loss.append(
                self.forward2d(output[:,:,f,:,:], target[:,:,f,:,:]))
        return sum(loss) / len(loss)


if __name__ == "__main__":
    for ch in [3, 1]:
        for cuda in [True, False]:
            batch_size, img_row, img_col = 32, 24, 24
            y_true = torch.rand(batch_size, ch, img_row, img_col)
            y_pred = torch.rand(batch_size, ch, img_row, img_col)
            if cuda:
                y_pred = y_pred.cuda()
                y_true = y_true.cuda()

            print('#'*20, 'Test on cuda : {} ; size : {}'.format(cuda, y_true.size()))

            y_pred_, y_true_ = y_pred.clone().requires_grad_(), y_true.clone()
            criterion = L1Loss()
            print('\t gradient bef : {}'.format(y_pred_.grad))
            loss = criterion(y_pred_, y_true_)
            loss.backward()
            print('\t gradient aft : {}'.format(y_pred_.grad.shape))
            print('{} : {}'.format(repr(criterion), loss.item()))

#             y_pred_, y_true_ = y_pred.clone().requires_grad_(), y_true.clone()
#             criterion = CELoss()
#             loss = criterion(y_pred_, y_true_)
#             loss.backward()
#             print('{} : {}'.format(repr(criterion), loss.item()))
#             print('\t gradient : {}'.format(y_pred_.grad.shape))

#             y_pred_, y_true_ = y_pred.clone().requires_grad_(), y_true.clone()
#             criterion = MSELoss()
#             loss = criterion(y_pred_, y_true_)
#             loss.backward()
#             print('{} : {}'.format(repr(criterion), loss.item()))
#             print('\t gradient : {}'.format(y_pred_.grad.shape))

#             y_pred_, y_true_ = y_pred.clone().requires_grad_(), y_true.clone()
#             criterion = FLoss()
#             loss = criterion(y_pred_, y_true_)
#             loss.backward()
#             print('{} : {}'.format(repr(criterion), loss.item()))
#             print('\t gradient : {}'.format(y_pred_.grad.shape))

#             y_pred_, y_true_ = y_pred.clone().requires_grad_(), y_true.clone()
#             criterion = VGG16Loss(cuda=cuda)
#             loss = criterion(y_pred_, y_true_)
#             loss.backward()
#             print('{} : {}'.format(repr(criterion), loss.item()))
#             print('\t gradient : {}'.format(y_pred_.grad.shape))

            y_pred_, y_true_ = y_pred.clone().requires_grad_(), y_true.clone()
            selector = NearestSelector()
            y_pred_near, y_true_near = selector.crop(y_pred_, y_true_)
            criterion = L1Loss()
            print('\t gradient bef : {}'.format(y_pred_.grad))
            loss = criterion(y_pred_near, y_true_near)
            loss.backward()
            print('\t gradient aft : {}'.format(y_pred_.grad.shape))
            print('{}-near : {}'.format(repr(criterion), loss.item()))

