#%%
import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from dataset import *
from model import UNet
from util import *

#%%
'''
random_b loss:  0.2659(random/test)/0.1837(d3)
sharpness_autocontrast loss: 0.0083(random2/test)/0.4814(d3)
근데 인터넷에서찾은건 random_b ㅇㅅㅇ(sharpness_autocontrast 트레인세트에 안드러감)
sharpness_autocontrast 50: 0.0095(random2/test)/0.4246(d3)
->sharpness_autocontrast 105: 0.0283(random2/test)/0.0360(d3) 
    ->1, 2 너무안나와서 그냥 삭제
autocontrast :  0.0062(random/test)/0.1808(d)
'''
#%%
'''
# parser
parser = argparse.ArgumentParser(description="Test the UNet",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--lr", default=1e-3,  type=float, dest="lr")
parser.add_argument("--batch_size", default=4, type=int, dest="batch_size")
parser.add_argument("--num_epoch", default=100, type=int, dest="num_epoch")
parser.add_argument("--data_dir", default="./train", type=str, dest="data_dir")
parser.add_argument("--ckpt_dir", default="./checkpoint", type=str, dest="ckpt_dir")
parser.add_argument("--result_dir", default="./result", type=str, dest="result_dir")

args = parser.parse_args()
lr = args.lr
batch_size = args.batch_size
num_epoch = args.num_epoch
data_dir = args.data_dir
ckpt_dir = args.ckpt_dir
result_dir =args.result_dir

print("learning rate: %.4e" % lr)
print("batch size: %d" % batch_size)
print("number of epoch: %d" % num_epoch)
print("data dir: %s" % data_dir)
print("ckpt dir: %s" % ckpt_dir)
print("result dir: %s" % result_dir)
'''
#%%
lr = 1e-3
batch_size =4
num_epoch = 100
img_dir ='/home/h/Desktop/data/dd/m_label'
label_dir ='/home/h/Desktop/data/dd/t_label'
ckpt_dir='/home/h/unet_pytorch_testing/autocontrast/ckpt'
result_dir ='/home/h/unet_pytorch_testing/autocontrast/eval_100_d'

print("learning rate: %.4e" % lr)
print("batch size: %d" % batch_size)
print("number of epoch: %d" % num_epoch)
print("img dir: %s" % img_dir)
print("label dir: %s" % label_dir)
print("ckpt dir: %s" % ckpt_dir)
print("result dir: %s" % result_dir)

# make folder if doesn't exist
if not os.path.exists(result_dir):
    os.makedirs(result_dir)

#%%
# network train
transform=transforms.Compose([transforms.Resize((512, 512)),
                              transforms.RandomAutocontrast(p=1),
                            #   transforms.RandomAdjustSharpness(sharpness_factor=2,p=1),
                              transforms.ToTensor(),
                              transforms.Normalize(mean=0.5, std=0.5)])
transform_label=transforms.Compose([transforms.Resize((512, 512)),
                              transforms.ToTensor(),
                              ])
test_dataset=CustomDataset(img_dir=img_dir,label_dir=label_dir,   transform=transform, transform_l=transform_label)       
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


# network generate
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
net = UNet().to(device)

# loss function, optimizer
fn_loss=nn.BCEWithLogitsLoss().to(device)
optim=torch.optim.Adam(net.parameters(), lr=lr)

# variables
num_data_test=len(test_dataset)

num_batch_test=np.ceil(num_data_test/batch_size)

# functions
fn_denorm=lambda x, mean, std:(x*std)+mean
fn_class=lambda x:1.0*(x>0.5) # network output image->binary class로 분류
tensor2PIL=transforms.ToPILImage()

#%%
# test network
net, optim, st_epoch = load(ckpt_dir=ckpt_dir, net=net, optim=optim)

with torch.no_grad(): # no backward pass 
    net.eval()
    loss_arr=[]

    for batch, data in enumerate(test_loader, 1):
        # forward pass
        input=data[0].to(device)
        label=data[1].to(device)
        output=net(input)
        
        # loss function
        loss = fn_loss(output, label)
        loss_arr+=[loss.item()]
        print("TEST: BATCH %04d / %04d | LOSS %.4f" %
                      (batch, num_batch_test, np.mean(loss_arr)))
        
        for i in range(input.shape[0]):
            inputimg=tensor2PIL(fn_denorm(input[i], mean=0.5, std=0.5))
            outputimg=tensor2PIL(fn_class(output[i]))
            bg= Image.open('transparence.png').resize((512, 512)) 
            bg.paste(inputimg,outputimg)
            
            name=data[2][i].split('/')[-1].replace('m_label', 'eval').replace('jpg','png')

            new_image = Image.new('RGB',(1024,512), (250,250,250))
            new_image.paste(inputimg,(0,0))
            new_image.paste(bg,(512,0))
            new_image.save(os.path.join(os.path.join(result_dir), name))
print("AVERAGE TEST: BATCH %04d / %04d | LOSS %.4f" %(batch, num_batch_test, np.mean(loss_arr)))

# %%
