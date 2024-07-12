import torch
import numpy as np
import scipy as sp
import skimage
import wandb
from tqdm import tqdm
from scipy.special import gamma
import os, sys
from PIL import Image
from losses import *
from networks import *
from ds import *
import random
random.seed(0)

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.optim
from torchvision import transforms, utils as tv_utils
from torchvision.transforms import functional as Fu

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark =True
dtype = torch.cuda.FloatTensor


def train_i2i_UNet3headGAN(
    netG_A,
    netD_A,
    train_loader, val_loader,
    experiment,
    ckpt_path,
    dtype=torch.cuda.FloatTensor,
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    resume_epoch=0,
    num_epochs=50,
    init_lr=1e-4,
    

):
    
    ####
    optimizerG = torch.optim.Adam(list(netG_A.parameters()), lr=init_lr)
    optimizerD = torch.optim.Adam(list(netD_A.parameters()), lr=init_lr)
    optimG_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizerG, num_epochs)
    optimD_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizerD, num_epochs)
    ####
    list_epochs = [50, 50, 150]
    list_lambda1 = [1, 0.5, 0.1]
    list_lambda2 = [0.0001, 0.001, 0.01]
    for num_epochs, lam1, lam2 in zip(list_epochs, list_lambda1, list_lambda2):
        for eph in range(resume_epoch, num_epochs):
            if eph%5 == 0  and eph > 0:
                print('\n ------------  Validation at Epoch :{} ------------\n'.format(eph))
                test_i2i_UNet3headGAN(netG_A, netD_A, lam1, lam2, val_loader, experiment, device, eph, dtype, ckpt_path)
            netG_A.train()
            netD_A.train()
            avg_D_loss   = 0
            avg_tot_loss = 0
            print('Number of train batches : ', len(train_loader))
            with tqdm(total=len(train_loader.dataset), desc=f'Epoch {eph}/{num_epochs}', unit="img") as pbar:
                for i, batch in enumerate(train_loader):
                    xA, xB = batch[0].to(device).type(dtype), batch[1].to(device).type(dtype)
                    #calc all the required outputs
                    rec_B, rec_alpha_B, rec_beta_B = netG_A(xA)

                    #first gen
                    netD_A.eval()
                    total_loss = lam1*F.l1_loss(rec_B, xB) + lam2*GndNll_loss(rec_B, rec_alpha_B, rec_beta_B, xB) + uarTV_loss(rec_beta_B) 
                    t0 = netD_A(rec_B)
                    t1 = F.avg_pool2d(t0, t0.size()[2:]).view(t0.size()[0], -1)
                    e5 = 0.001*F.mse_loss(t1, torch.ones(t1.size()).to(device).type(dtype))  # Loss for generator, target 1 for rec_B.
                    total_loss += e5
                    optimizerG.zero_grad()
                    total_loss.backward()
                    optimizerG.step()
    
                    #then discriminator
                    netD_A.train()
                    t0 = netD_A(xB)
                    pred_real_A = F.avg_pool2d(t0, t0.size()[2:]).view(t0.size()[0], -1)
                    loss_D_A_real = 1*F.mse_loss(
                        pred_real_A, torch.ones(pred_real_A.size()).to(device).type(dtype)
                    )
                    t0 = netD_A(rec_B.detach())
                    pred_fake_A = F.avg_pool2d(t0, t0.size()[2:]).view(t0.size()[0], -1)
                    loss_D_A_pred = 1*F.mse_loss(
                        pred_fake_A, torch.zeros(pred_fake_A.size()).to(device).type(dtype)
                    )
                    loss_D_A = (loss_D_A_real + loss_D_A_pred)*0.5
    
                    loss_D = loss_D_A
                    optimizerD.zero_grad()
                    loss_D.backward()
                    optimizerD.step()
    
                    avg_tot_loss += total_loss.item()
                    avg_D_loss += loss_D.item()                    
                    
                    ### progress bar
                    pbar.update(xA.shape[0])
                    pbar.set_postfix(**{'loss (batch)': total_loss.item()})
                    
                    
                avg_tot_loss /= len(train_loader)
                avg_D_loss /= len(train_loader)
                print(
                    '\n epoch: [{:2d}/{:2d}] | avg_tot_loss: {:.4f} \n'.format(
                        eph, num_epochs, avg_tot_loss)
                    )
                    
                ### Save weights
                torch.save(netG_A.state_dict(), ckpt_path+'_eph{}_G_A.pth'.format(eph))
                torch.save(netD_A.state_dict(), ckpt_path+'_eph{}_D_A.pth'.format(eph))
                    
        return netG_A, netD_A

def test_i2i_UNet3headGAN(
                        netG_A,
                        netD_A,
                        lam1,
                        lam2,
                        test_loader,
                        experiment,
                        device,
                        eph,
                        dtype,
                        ckpt_path
                        ):
    netG_A.eval()
    netD_A.eval()
    avg_loss_D = 0
    avg_tot_loss = 0
    for i, batch in tqdm(enumerate(test_loader), unit="batch", total=len(test_loader)):
        xA, xB = batch[0].to(device).type(dtype), batch[1].to(device).type(dtype)
        #calc all the required outputs 
        rec_B, rec_alpha_B, rec_beta_B = netG_A(xA)
    
        # Generator
        total_loss = lam1*F.l1_loss(rec_B, xB) + lam2*GndNll_loss(rec_B, rec_alpha_B, rec_beta_B, xB) + uarTV_loss(rec_beta_B)
        t0 = netD_A(rec_B)
        t1 = F.avg_pool2d(t0, t0.size()[2:]).view(t0.size()[0], -1)
        e5 = 0.001*F.mse_loss(t1, torch.ones(t1.size()).to(device).type(dtype))  # Loss for generator, target 1 for rec_B.
        total_loss += e5
        
        # Discriminator
        t0 = netD_A(xB)
        pred_real_A = F.avg_pool2d(t0, t0.size()[2:]).view(t0.size()[0], -1)
        loss_D_A_real = 1*F.mse_loss(
                                    pred_real_A, torch.ones(pred_real_A.size()).to(device).type(dtype)
                                    )
        t0 = netD_A(rec_B.detach())
        pred_fake_A = F.avg_pool2d(t0, t0.size()[2:]).view(t0.size()[0], -1)
        loss_D_A_pred = 1*F.mse_loss(
                                   pred_fake_A, torch.zeros(pred_fake_A.size()).to(device).type(dtype)
                                   )
        loss_D_A = (loss_D_A_real + loss_D_A_pred)*0.5
    
        avg_tot_loss += total_loss.item()
        avg_loss_D += loss_D_A.item()
        
    
    avg_tot_loss /= len(test_loader)
    avg_loss_D /=len(test_loader)
    
    print('/n=============== Validation ==================/n')
    print('epoch: [{:2d}] | avg_tot_loss: [{:.5f}] | avg_D_loss: [{:.5f}]'.format(
                    eph, avg_tot_loss, avg_loss_D))
    