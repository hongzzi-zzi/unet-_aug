#%%
import argparse
import os
import warnings

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from dataset import *
from model import UNet
from util import load

%matplotlib inline 
import matplotlib.pyplot as plt

warnings.filterwarnings(action='ignore')
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
test_dir = '/home/h/Desktop/data/random2/test/m_label' ## 이거 transform 해서 테스트해보깅
ckpt_dir = '/home/h/unet_pytorch_testing/sharpness_autocontrast/ckpt'
result_dir ='/home/h/unet_pytorch_testing/sharpness_autocontrast/result'
test_lst=os.listdir(test_dir)
print("learning rate: %.4e" % lr)
print("batch size: %d" % batch_size)
print("test dir: %s" % test_dir)
print("ckpt dir: %s" % ckpt_dir)
print("result dir: %s" % result_dir)


#%%

# make folder if doesn't exist
if not os.path.exists(os.path.join(result_dir,'mask')):
    os.makedirs(os.path.join(result_dir,'mask'))
if not os.path.exists(os.path.join(result_dir,'compare')):
    os.makedirs(os.path.join(result_dir,'compare'))
    print('make new result_dir')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#%%
## 한번 쭉 보정한다음에 넣으,면 더 잘 알아들을까 ?ㅇㅅㅇ
fn_denorm=lambda x, mean, std:(x*std)+mean
fn_class = lambda x: 1.0 * (x > 0.5)
tensor2PIL=transforms.ToPILImage()
# transform=transforms.Compose([transforms.Resize((512, 512)),
#                               transforms.ToTensor(),
#                               transforms.Normalize(mean=0.5, std=0.5)])
transform=transforms.Compose([transforms.Resize((512, 512)),
                              transforms.RandomAutocontrast(p=1),
                              transforms.RandomAdjustSharpness(sharpness_factor=2,p=1),
                              transforms.ToTensor(),
                              transforms.Normalize(mean=0.5, std=0.5)])
# transform=transforms.Compose([transforms.Resize((512, 512)),
#                               transforms.RandomHorizontalFlip(), 
#                               transforms.RandomVerticalFlip(), 
#                               transforms.RandomAffine([-60, 60]),
#                               transforms.ColorJitter(brightness=0.5, contrast=0.5),
#                               transforms.ToTensor(),
#                               transforms.Normalize(mean=0.5, std=0.5)
#                               ])
test_dataset=CustomDataset(img_dir=test_dir,  transform=transform)       
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

#%%
net = UNet().to(device)
fn_loss=nn.BCEWithLogitsLoss().to(device)
optim=torch.optim.Adam(net.parameters(), lr=lr)
net, optim, st_epoch = load(ckpt_dir=ckpt_dir, net=net, optim=optim)
#%%
with torch.no_grad(): # no backward pass 
    net.eval()
    for batch, data in enumerate(test_loader, 1):
        input=data[0].to(device)
        output=net(input)
        
        for i in range(input.shape[0]):
            inputimg=tensor2PIL(fn_denorm(input[i], mean=0.5, std=0.5))
            outputimg=tensor2PIL(fn_class(output[i]))
            bg= Image.open('transparence.png').resize((512, 512)) 
            bg.paste(inputimg,outputimg)
            # print(data[1][i]) #/home/h/Desktop/data/random2/test/m_label/m_label5-5_109.png
            name=data[1][i].split('/')[-1].replace('m_label', 't_output').replace('jpg','png')
            # outputimg.save(os.path.join(os.path.join(result_dir,'mask'), name))
            
            new_image = Image.new('RGB',(1024,512), (250,250,250))
            new_image.paste(inputimg,(0,0))
            new_image.paste(bg,(512,0))
            new_image.save(os.path.join(os.path.join(result_dir,'compare'), name.replace('t_output', 'comp').replace('jpg','png')))
#%%
