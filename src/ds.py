import torch.utils.data as data
import os.path
import glob
import numpy as np
import random
import torch
from torch import Tensor
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode as IMode

from PIL import Image

random.seed(0)

import utils

class add_gaussian_noise(object):
    def __init__(self, mean=0.0, var=0.01, p=0.0):
        self.mean = mean
        self.var = var
        self.p = p
    def __call__(self, img):
        if random.uniform(0, 1) < self.p:
            std = self.var**0.5
            image_array = np.array(img)
            noisy_img = image_array + np.random.normal(self.mean, std, image_array.shape)
            return torch.from_numpy(noisy_img)
        else:
            return img
        
    

class Paired_dataset(Dataset):
    '''
    can act as supervised or un-supervised based on flists
    '''
    def __init__(self, flist1, flist2, image_size):
        
        self.flist1 = flist1
        self.flist2 = flist2
        self.transform = transforms.Compose([
                transforms.CenterCrop(image_size)
                ])

    def __getitem__(self, index):
        impath1 = self.flist1[index]
        img1 = Image.open(impath1).convert('RGB')
        impath2 = self.flist2[index]
        img2 = Image.open(impath2).convert('RGB')
        
        img1 = utils.image2tensor(img1, range_norm=False, half=False)
        img2 = utils.image2tensor(img2, range_norm=False, half=False)
        
        if self.transform is not None:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
        
        return img1, img2
    def __len__(self):
        return len(self.flist1)

def call_paired(train, image_size, test_mode=None):
    if train:
        root1='../data/train/orig'
        root2='../data/train/fice'
    else:
        root1='../data/test/orig'
        root2='../data/test/fice'

    flist1=glob.glob(root1+'/*.jpg')
    flist2=glob.glob(root2+'/*.jpg')

    dset = Paired_dataset(flist1,flist2,image_size)
    return dset
