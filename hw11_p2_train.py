workspace_dir = '.'

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

class WDis(nn.Module):
    """
    input (N, 3, 64, 64)
    output (N, )
    """
    def __init__(self, in_dim, dim=64):
        super(WDis, self).__init__()
        
        self.cnn = nn.Sequential(
            nn.Conv2d(in_dim, dim, 4, 2, 1, bias=False), # (N, dim, 32, 32)
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(dim, dim*2, 4, 2, 1, bias=False), # (N, dim*2, 16, 16)
            nn.BatchNorm2d(dim*2),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(dim*2, dim*4, 4, 2, 1, bias=False), # (N, dim*4, 8, 8)
            nn.BatchNorm2d(dim*4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(dim*4, dim*8, 4, 2, 1, bias=False), # (N, dim*8, 4, 4)
            nn.BatchNorm2d(dim*8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(dim*8, 1, 4, 1, 0, bias=False), # (N, 1, 1, 1)
            # Modification 1: remove sigmoid
            # nn.Sigmoid()
        )
        self.apply(weights_init)        
    def forward(self, x):
        y = self.cnn(x)
        y = y.view(-1)
        return y

# hyperparameters 
batch_size = 64
z_dim = 100
lr = 1e-4
n_epoch = 10
save_dir = os.path.join(workspace_dir, 'logs')
os.makedirs(save_dir, exist_ok=True)

# model
G = WGen(in_dim=z_dim).cuda()
D = WDis(3).cuda()
G.train()
D.train()

# loss criterion
criterion = nn.BCELoss()

# modification 2: Use RMSprop instead of Adam
# optimizer
opt_D = torch.optim.RMSprop(D.parameters(), lr=lr)
opt_G = torch.optim.RMSprop(G.parameters(), lr=lr)


same_seeds(0)
# dataloader (You might need to edit the dataset path if you use extra dataset.)
dataset = get_dataset(os.path.join(sys.argv[1]))
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

# for logging
z_sample = Variable(torch.randn(100, z_dim)).cuda()

for e, epoch in enumerate(range(n_epoch)):
    for i, data in enumerate(dataloader):
        imgs = data
        imgs = imgs.cuda()

        bs = imgs.size(0)

        # modification: clip param for discriminator
        for parm in D.parameters():
            parm.data.clamp_(-clamp_num, clamp_num)

        """ Train D """
        z = Variable(torch.randn(bs, z_dim)).cuda()
        r_imgs = Variable(imgs).cuda()
        f_imgs = G(z)

        # label        
        r_label = torch.ones((bs)).cuda()
        f_label = torch.zeros((bs)).cuda()

        # dis
        r_logit = D(r_imgs.detach())
        f_logit = D(f_imgs.detach())
        
        # # compute loss
        # r_loss = criterion(r_logit, r_label)
        # f_loss = criterion(f_logit, f_label)
        # loss_D = (r_loss + f_loss) / 2
        V_D = np.sum(r_logit.cpu().data.numpy()) - np.sum(f_logit.cpu().data.numpy()) # to maximize

        # update model
        D.zero_grad()
        r_logit.backward(r_label)
        f_logit.backward(r_label*(-1))
        # loss_D.backward()
        opt_D.step()

        """ train G """
        # leaf
        z = Variable(torch.randn(bs, z_dim)).cuda()
        f_imgs = G(z)

        # dis
        f_logit = D(f_imgs)
        
        # compute loss
        # loss_G = criterion(f_logit, r_label)
        V_G = - np.sum(f_logit.cpu().data.numpy()) # to minimize

        # update model
        G.zero_grad()
        # loss_G.backward()
        f_logit.backward(r_label)
        opt_G.step()

        # log
        # print(f'\rEpoch [{epoch+1}/{n_epoch}] {i+1}/{len(dataloader)} Loss_D: {loss_D.item():.4f} Loss_G: {loss_G.item():.4f}', end='')
        print(f'\rEpoch [{epoch+1}/{n_epoch}] {i+1}/{len(dataloader)} Loss_D: {V_D:.4f} Loss_G:{V_G:.4f}', end='')
    G.eval()
    f_imgs_sample = (G(z_sample).data + 1) / 2.0
    filename = os.path.join(save_dir, f'Epoch_{epoch+1:03d}.jpg')
    torchvision.utils.save_image(f_imgs_sample, filename, nrow=10)
    print(f' | Save some samples to {filename}.')
    G.train()
    if (e+1) % 5 == 0:
        torch.save(G.state_dict(), sys.argv[2])
