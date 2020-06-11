# Code from https://github.com/chenyuntc/pytorch-GAN/blob/master/WGAN.ipynb

import sys
import torch
from torch import nn
from torch.autograd import Variable
from torch import optim
import torchvision
from torchvision import transforms
from torchvision.utils import make_grid
# from torchvision.datasets import CIFAR10
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from pylab import plt
import os
import cv2

lr = 0.00005
nz = 100 # noise dimension
image_size = 64
image_size2 = 64
nc = 3 # chanel of img 
ngf = 64 # generate channel
ndf = 64 # discriminative channel
beta1 = 0.5
batch_size = 32
max_epoch = 50 # =1 when debug
workers = 2
gpu = True # use gpu or not
clamp_num=0.01# WGAN clip gradient

# # data preprocess

class FaceDataset(Dataset):
    def __init__(self, fnames, transform):
        self.transform = transform
        self.fnames = fnames
        self.num_samples = len(self.fnames)
    def __getitem__(self,idx):
        fname = self.fnames[idx]
        img = cv2.imread(fname)
        img = self.BGR2RGB(img) #because "torchvision.utils.save_image" use RGB
        img = self.transform(img)
        return img

    def __len__(self):
        return self.num_samples

    def BGR2RGB(self,img):
        return cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

import glob

def get_dataset(root):
    fnames = glob.glob(os.path.join(root, '*'))
    # resize the image to (64, 64)
    # linearly map [0, 1] to [-1, 1]
    transform = transforms.Compose(
        [transforms.ToPILImage(),
         transforms.Resize((64, 64)),
         transforms.ToTensor(),
         transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3) ] )
    dataset = FaceDataset(fnames, transform)
    return dataset

# # Utils

import random
import torch
import numpy as np

def same_seeds(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

# # Define Model

def weights_init(m):
    # weight_initialization: important for wgan
    classname=m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

class WGen(nn.Module):
    """
    input (N, in_dim)
    output (N, 3, 64, 64)
    """
    def __init__(self, in_dim, dim=64):
        super(WGen, self).__init__()
        
        self.cnn = nn.Sequential(
            nn.ConvTranspose2d(in_dim, dim*8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(dim*8),
            nn.ReLU(True),

            nn.ConvTranspose2d(dim*8, dim*4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(dim*4),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(dim*4, dim*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(dim*2),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(dim*2, dim, 4, 2, 1, bias=False),
            nn.BatchNorm2d(dim),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(dim, 3, 4, 2, 1, bias=False),
            nn.Tanh(),
        )
        self.apply(weights_init)

    def forward(self, x):
        y = x.view(x.size(0), -1, 1, 1) # (batch size, input dim, 1, 1)
        y = self.cnn(y)
        return y

# for logging
z_dim = 100
z_sample = Variable(torch.randn(100, z_dim)).cuda()

# # Inference
# 利用我們訓練好的 Generator 來隨機生成圖片。

import torch
# load pretrained model
G = WGen(z_dim)
G.load_state_dict(torch.load(sys.argv[1]))
G.eval()
G.cuda()


# generate images and save the result
same_seeds(1)
n_output = 20
z_sample = Variable(torch.randn(n_output, z_dim)).cuda()
imgs_sample = (G(z_sample).data + 1) / 2.0
torchvision.utils.save_image(imgs_sample, sys.argv[2], nrow=10)
