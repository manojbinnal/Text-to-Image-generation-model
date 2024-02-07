
#Group Project
#import the libraries

import torch,pdb
import torch.nn as nn
from torch.utils.data import DataLoader
from torch import nn
from torchvision import transforms
from torchvision.datasets import MNIST
from torchvision.utils import make_grid
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

#visualization function
def show(tensor,ch=1,size=(28,28),num=16):
  #tensor:128 X
  data=tensor.detach().cpu().view(-1,ch,*size) # 128 x 1 x 28 x 28
  grid= make_grid(data[:num],nrows=4).permute(1,2,0) #1 x 28 x 28 = 28 x 28 x 1
  plt.inshow(grid)
  plt.show()

# setup of the main parameters and hyperparameters
epochs=500
cur_step=0
info_step=300
mean_gen_loss=0
mean_disc_loss=0

z_dim=64
lr = 0.00001 #learning rate
loss_func=nn.BCEWithLogitsLoss()

bs=128   #batch size
device='cuda'

dataloader =DataLoader(MNIST('.',download=True,transform=transforms.ToTensor()),shuffle=True,batch_size=bs)

# declare our models

# generator
def genBlock(inp,out): # input and output size
  return nn.Sequential(
    nn.Linear(inp,out),
    nn.BatchNorm1d(out),
    nn.ReLU(inplace=True)
  )

class Generator(nn.Module):
  def __init__(self,z_dim,i_dim=784,h_dim=128):
    super().__init__()
    self.gen=nn.Sequential(
        genBlock(z_dim,h_dim,), #
        genBlock(h_dim,h_dim*2),
        genBlock(h_dim*2,h_dim*4),
        genBlock(h_dim*4,h_dim*8),
        nn.Linear(h_dim*8,i_dim),
        nn.Sigmoid()
    )
    def forward(self,noise):
      return self.gen(noise)

def gen_noise(number,z_dim):
  return torch.randn(number,z_dim).to(device)