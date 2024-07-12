#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 21 10:40:37 2024

@author: user1
"""


import numpy as np
import random
import wandb
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode as IMode

from PIL import Image

from ds import *
from losses import *
from networks import *
from utils import *

dtype = torch.cuda.FloatTensor

# Initialize Dataset
ImgDset = call_paired(train = True, image_size = 490)
testDset = call_paired(train = False, image_size = 490)

# Reducing training dtaaset size, as currently it takes too much time for the problem.
n_train, n_val = 5000, 5000
train_dset, val_dset = random_split(ImgDset, [n_train, n_val])
# Initialize loaders
train_loader = DataLoader(train_dset, batch_size=4, pin_memory=False, shuffle=True)
val_loader = DataLoader(val_dset, batch_size=4, pin_memory=False, shuffle=False)
test_loader = DataLoader(testDset, batch_size=4, pin_memory=False, shuffle=False)

# Initialize models
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

netG_A = UNet_3head(3,3)
netG_A = nn.DataParallel(netG_A, device_ids=[0,1])
netG_A.to(device)
netG_A.type(dtype)
####
netD_A = NLayerDiscriminator(3, n_layers=4)
netD_A = nn.DataParallel(netD_A, device_ids=[0,1])
netD_A.to(device)
netD_A.type(dtype)

#Load saved model weights
netG_A.load_state_dict(torch.load('../ckpt/UAR/uar_eph49_G_A.pth', map_location='cuda'))
netD_A.load_state_dict(torch.load('../ckpt/UAR/uar_eph49_D_A.pth', map_location='cuda'))
    
# Model parameters
model_parameters = filter(lambda p: True, netG_A.parameters())
params = sum([np.prod(p.size()) for p in model_parameters])
print("Number of Parameters in G:",params)

### Initialize logging ###
logger = wandb.init(project='I2I', resume='allow', tags=["UAR"])


# Train
netG_A, netD_A = train_i2i_UNet3headGAN(
    netG_A, netD_A,
    train_loader, val_loader,
    experiment = logger,
    ckpt_path='../ckpt/UAR/',
    dtype=torch.cuda.FloatTensor,
    device = device,
    resume_epoch = 0,
    num_epochs=50,
    init_lr=1e-4,

)
