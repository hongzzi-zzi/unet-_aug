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

#%%
# functions

def load_image_resize_withoutalpha(imfile, new_w, new_h):
    aaa=Image.open(imfile).convert('RGB').resize((new_w, new_h), Image.BILINEAR)
    img = np.array(aaa).astype(np.uint8)
    return img

fn_totensor=lambda x: torch.from_numpy(x.transpose((2,0,1)).astype(np.float32))
fn_tonumpy=lambda x:x.to('cpu').detach().numpy().transpose(0, 2, 3, 1)
fn_norm = lambda x, mean, std: (x-mean)/std
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
        input=data[0].to(device).unsqueeze(0)
        # print(input.shape)# torch.Size([1, 3, 512, 512])
        output=net(input)# torch.Size([1, 3, 512, 512])
        output=fn_tonumpy(fn_class(output))# (1, 512, 512, 1)
        input=fn_tonumpy(fn_denorm(input, mean=0.5, std=0.5))# (4, 512, 512, 3)
        compout=np.zeros((output.shape[0], output.shape[1], output.shape[2], 3))
        for each_channel in range(3):
                compout[:,:,:,each_channel] = output[:,:,:,0]
        comparray=np.hstack((compout,input))
        comp=Image.fromarray(np.uint8(comparray.squeeze()*225))
        test_file=os.path.join(test_dir,sorted(os.listdir(test_dir))[idx])
        nameee='comp'+str(idx)+'.png'
        idx+=1
        
        comp.resize((512, 1024)).save(os.path.join(result_dir, nameee))
        ## 저장하는거 만들기

#%%

#%%
'''for test_file in test_lst:
    test_file=os.path.join(test_dir,test_file)
    if os.path.isfile(test_file):
        ### 파일 열 떄 자동보정해서 넣을수 없나 ㅇㅅㅇ?

        # 255로 나누는 이유 : 이미지 값의 범위를 0~255에서 0~1 값의 범위를 갖도록 하기 위함
        input=load_image_resize_withoutalpha(test_file, 512, 512)/255.0# (512, 512, 3), ndim=3
        if input.ndim == 2:
            input = input[:, :, np.newaxis]

        input=fn_totensor(fn_norm(input, 0.5, 0.5)).to(device)# torch.Size([3, 512, 512])
        input=input.unsqueeze(0)# torch.Size([1, 3, 512, 512])

        net = UNet().to(device)
        fn_loss=nn.BCEWithLogitsLoss().to(device)
        optim=torch.optim.Adam(net.parameters(), lr=lr)

        net, optim, st_epoch = load(ckpt_dir=ckpt_dir, net=net, optim=optim)

        with torch.no_grad(): # no backward pass 
            net.eval()
            output=net(input)# torch.Size([1, 3, 512, 512])
            output=fn_tonumpy(fn_class(output))# (1, 512, 512, 1)
            input=fn_tonumpy(fn_denorm(input, mean=0.5, std=0.5))# (4, 512, 512, 3)
            compout=np.zeros((output.shape[0], output.shape[1], output.shape[2], 3))
            for each_channel in range(3):
                compout[:,:,:,each_channel] = output[:,:,:,0]
            comparray=np.hstack((compout,input))
            comp=Image.fromarray(np.uint8(comparray.squeeze()*225))
            nameee='comp_'+test_file.split('_')[-1]
            
            comp.resize((512, 1024)).save(os.path.join(result_dir, nameee))
            
        # output=Image.fromarray(np.uint8(output.squeeze()*225))
        # output.resize(origin_size).save(os.path.join(result_dir, name))'''
# %%
