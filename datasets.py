#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
  @Email: guengmingwu2010@gmail.com
  @Copyright: go-hiroaki
  @License: MIT
"""

import argparse
import os
import numpy as np
import pandas as pd
from PIL import Image
from skimage.io import imsave
from skimage.color import lab2rgb, rgb2lab, rgb2gray

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from transform import *

import warnings
warnings.filterwarnings("ignore")

src_DIR = os.pathdirname(os.path.abspath(__file__))
Dataset_DIR = os.path.join(Src_DIR, '../datasets/')


def showPixelRange(arr, channels = ['R', 'G', 'B']):
  """
  args:
    arr: 3d img array
    channels: list of channel name
  return 0
  """
  assert arr.shape[-1] == len(channels), "Channel should be consistent."
  print("{} Image:".format(''.join(channels)))
  for idx, ch in enumerate(channels):
    ch_max, ch_min = np.max(arr[:,:,idx]), np.min(arr[:,:,idx])
    print('\t{}-Channel : Max:{} ; Min:{} ;".format(ch, ch_max, ch_min))
          
          
