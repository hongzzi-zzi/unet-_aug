#%%
import argparse
import os
import warnings

import numpy as np
import torch
import torch.nn as nn
from PIL import Image


from model import UNet
from util import load
from dataset import *
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
%matplotlib inline 
import matplotlib.pyplot as plt

warnings.filterwarnings(action='ignore')

#%% functions
fn_denorm=lambda x, mean, std:(x*std)+mean
fn_class = lambda x: 1.0 * (x > 0.5)
#%% parser
'''parser = argparse.ArgumentParser(description="read png",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--lr", default=1e-3,  type=float)
parser.add_argument("--batch_size", default=4, type=int)
parser.add_argument("--test_dir", type=str)
parser.add_argument("--ckpt_dir", type=str)
parser.add_argument("--result_dir", type=str)

args = parser.parse_args()
lr = args.lr
batch_size = args.batch_size
test_dir = args.test_dir
ckpt_dir = args.ckpt_dir
result_dir =args.result_dir
test_lst=os.listdir(test_dir)'''

#%%
lr = 1e-3
batch_size = 4
test_dir = '/home/h/Desktop/data/random_test/m_label'
ckpt_dir = 'random_train/ckpt'
result_dir ='random_train/result'
test_lst=os.listdir(test_dir)
print("learning rate: %.4e" % lr)
print("batch size: %d" % batch_size)
print("test dir: %s" % test_dir)
print("ckpt dir: %s" % ckpt_dir)
print("result dir: %s" % result_dir)

# make folder if doesn't exist
if not os.path.exists(result_dir):
    os.makedirs(result_dir)
    print('make new result_dir')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#%%
## 한번 쭉 보정한다음에 넣으,면 더 잘 알아들을까 ?ㅇㅅㅇ
transform=transforms.Compose([transforms.Resize((512, 512)),
                              transforms.ToTensor(),
                              transforms.Normalize(mean=0.5, std=0.5)])
test_dataset=CustomDataset(img_dir=test_dir,  transform=transform)       
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

#%%
net = UNet().to(device)
fn_loss=nn.BCEWithLogitsLoss().to(device)
optim=torch.optim.Adam(net.parameters(), lr=lr)
net, optim, st_epoch = load(ckpt_dir=ckpt_dir, net=net, optim=optim)
#%%
tensor2PIL=transforms.ToPILImage()
idx=0
with torch.no_grad(): # no backward pass 
    net.eval()
    for batch, data in enumerate(test_loader, 1):
        input=data[0].to(device)# torch.Size([1, 3, 512, 512])
        output=net(input)# torch.Size([1, 3, 512, 512])
        for i in range(input.shape[0]):
            inputimg=tensor2PIL(fn_denorm(input[i], mean=0.5, std=0.5))
            outputimg=tensor2PIL(fn_class(output[i]))
            
            name=data[1][i].split('/')[-1].replace('m_label', 't_output')
            new_image = Image.new('RGB',(1024,512), (250,250,250))
            new_image.paste(inputimg,(0,0))
            new_image.paste(outputimg,(512,0))
            new_image.save(os.path.join(result_dir, name.replace('t_output', 'comp')),'png')
            outputimg.save(os.path.join(result_dir, name),'png')
#%%