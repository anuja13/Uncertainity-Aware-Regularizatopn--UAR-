from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import lpips
import torch
import numpy as np
from matplotlib import pyplot as plt
import scipy.ndimage.filters as fi
import matplotlib.pyplot as plt

def compare_ssim(imgRef, imgT, K1=0.01, K2=0.03):
    r = ssim(imgRef, imgT, data_range=imgT.max() - imgT.min(), channel_axis=0, K1=K1, K2=K2)
    return r

def compare_psnr(imgRef, imgT):
    r = psnr(imgRef, imgT, data_range=imgT.max() - imgT.min())
    return r

def lpips_loss():
    lpips_metric = lpips.LPIPS(net='vgg')
    return lpips_metric

def compare_rrmse(imgRef, imgT):
    numerator = (imgRef-imgT)**2
    numerator = np.mean(numerator.flatten())
    
    denominator = (imgRef)**2
    denominator = np.mean(denominator.flatten())
    
    r = numerator/denominator
    r = np.sqrt(r)
    return r


def GndNll_loss(out_mean, out_1alpha, out_beta, target):
    alpha_eps, beta_eps = 1e-5, 1e-5
    out_1alpha += alpha_eps
    out_beta += beta_eps 
    factor = out_1alpha
    resi = torch.abs(out_mean - target)

    resi = (resi*factor*out_beta).clamp(min=1e-4, max=50)
    log_1alpha = torch.log(out_1alpha)
    log_beta = torch.log(out_beta)
    lgamma_beta = torch.lgamma(torch.pow(out_beta, -1))
    
    if torch.sum(log_1alpha != log_1alpha) > 0:
        print('log_1alpha has nan')
        print(lgamma_beta.min(), lgamma_beta.max(), log_beta.min(), log_beta.max())
    if torch.sum(lgamma_beta != lgamma_beta) > 0:
        print('lgamma_beta has nan')
    if torch.sum(log_beta != log_beta) > 0:
        print('log_beta has nan')
    
    l = resi - log_1alpha + lgamma_beta - log_beta
    l = torch.mean(l)
    return l


def save_model(M, M_ckpt):
    torch.save(M.state_dict(), M_ckpt)
    print('model saved @ {}'.format(M_ckpt))

def uarTV_loss(img, eph):
    """
    Compute total variation loss.
    
    Inputs:
    - img: PyTorch Variable of shape (1, 3, H, W) holding an input image.
    - tv_weight: Scalar giving the weight w_t to use for the TV loss.
    
    Returns:
    - loss: PyTorch Variable holding a scalar giving the total variation loss
      for img weighted by tv_weight.
    """

    eps = 1e-7
    lambd = 1e-12

    w_variance = torch.sum(torch.pow(img[:,:,:,:-1] - img[:,:,:,1:], 2))
    h_variance = torch.sum(torch.pow(img[:,:,:-1,:] - img[:,:,1:,:], 2))
    
    if eph < 5:
        loss = 0
    else:
        loss = lambd * torch.pow((h_variance + w_variance + eps), 0.5)

    return loss
        
def compute_quality(rec_B, xB):
    psnr = []
    ssim = []
    rrmse = []
    lpips_metric = lpips_loss().cuda()
    # # calculate numeric metrics
    for i in range(xB.shape[0]):
        rec, x = rec_B[i], xB[i]
        psnr_val= compare_psnr(x.detach().cpu().numpy(), rec.detach().cpu().numpy())
        ssim_val = compare_ssim(x.detach().cpu().numpy(), rec.detach().cpu().numpy())
        rrmse_val = compare_rrmse(x.detach().cpu().numpy(), rec.detach().cpu().numpy())
        psnr.append(psnr_val)
        ssim.append(ssim_val)
        rrmse.append(rrmse_val)
    
    psnr_avg = sum(psnr)/len(psnr)
    ssim_avg = sum(ssim)/len(ssim)
    rrmse_avg =sum(rrmse)/len(rrmse)
    
    psnr.append(psnr_avg)
    ssim.append(ssim_avg)
    
    
    #lpips
    images = torch.clamp(xB.data, -1, 1)
    recons = torch.clamp(rec_B.data, -1, 1)
    lpips_score = lpips_metric(images, recons)
    lpips_score = torch.sum(lpips_score)/lpips_score.shape[0]
    
    return lpips_score, psnr_avg, ssim_avg, rrmse_avg, psnr, ssim

