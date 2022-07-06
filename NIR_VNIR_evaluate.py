#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
  @Email:  guangmingwu2010@gmail.com
  @Copyright: go-hiroaki
  @License: MIT
"""
import os
import cv2
import PIL
import glob
import random
import torchvision
import numpy as np
import utils
from skimage.io import imread, imsave
import argparse
import csv
import pandas as pd

data = pd.read_csv('/media/kimhyeongyu/50AEDF33AEDF0FF8/VSIAD/src/result/raw/resnetpix2pixbeta6-VC24-RGB2LAB.csv')
data_NIR = data[(data['scene'] == 'NIR')]
data_VNIR = data[(data['scene'] == 'VNIR')]

print('NIR-PSNR ==> {}' .format(data_NIR['PSNR'].sum() / len(data_NIR)))
print('NIR-SSIM ==> {}' .format(data_NIR['SSIM'].sum() / len(data_NIR)))
print('NIR-AE ==> {}' .format(data_NIR['AE'].sum() / len(data_NIR)))
print('NIR-LPIPS ==> {}' .format(data_NIR['LPIPS'].sum() / len(data_NIR)))
print('NIR-FID ==> {}' .format(data_NIR['FID'].sum() / len(data_NIR)))



print('VNIR-PSNR ==> {}' .format(data_VNIR['PSNR'].sum() / len(data_VNIR)))
print('VNIR-SSIM ==> {}' .format(data_VNIR['SSIM'].sum() / len(data_VNIR)))
print('VNIR-AE ==> {}' .format(data_VNIR['AE'].sum() / len(data_VNIR)))
print('VNIR-LPIPS ==> {}' .format(data_VNIR['LPIPS'].sum() / len(data_VNIR)))
print('VNIR-FID ==> {}' .format(data_VNIR['FID'].sum() / len(data_VNIR)))

