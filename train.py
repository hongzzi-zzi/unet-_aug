#%%
import argparse
import os
from tabnanny import verbose

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms

from dataset import *
from model import UNet
from util import *
#%%
'''
    모든게 trainset 1 2 윗니 잘 안나옴 ㅇㅅㅇ...
    
    random3: 전체data 90 train->1 윗니 잘안나옴(100)
    random_train copy: 그냥랜덤(epoch 100)+train continue(원래꺼+ 5-3, 5-4 반쯤만 추가한 trainset, 150)-> 1 윗니 잘안나옴
    random_train copy 2:그냥랜덤(epoch 100)+train continue(5-3, 5-4 반쯔만, 150)-> 1윗니 잘안나옴
    
    -->random3 +trainset1(ckpt이어서 130번)해도 앞니 잘안나오는거 이씀
    이 커야 나옴
    
    random4:1, 2 crop epoch 100->뭔가 잘못됨...야발
'''
#%%
'''# # parser
# parser = argparse.ArgumentParser(description="Train the UNet",
#                                  formatter_class=argparse.ArgumentDefaultsHelpFormatter)

# parser.add_argument("--lr", default=1e-3,  type=float, dest="lr")
# parser.add_argument("--batch_size", default=4, type=int, dest="batch_size")
# parser.add_argument("--num_epoch", default=100, type=int, dest="num_epoch")

# parser.add_argument("--data_dir", default="./train", type=str, dest="data_dir")
# parser.add_argument("--ckpt_dir", default="./checkpoint", type=str, dest="ckpt_dir")
# parser.add_argument("--log_dir", default="./log", type=str, dest="log_dir")
# parser.add_argument("--train_continue", default="off", type=str, dest="train_continue")

# args = parser.parse_args()

# # training parameter
# lr = args.lr
# batch_size = args.batch_size
# num_epoch = args.num_epoch
# data_dir = args.data_dir
# ckpt_dir = args.ckpt_dir
# log_dir = args.log_dir
# train_continue = args.train_continue'''
#%%
# training parameter
lr = 1e-3
batch_size = 4 # 6이 최대
num_epoch = 100
ckpt_dir = 'random/ckpt'
log_dir = 'random/log'
train_continue = 'on'
## dlalwl
img_dir='/home/h/Desktop/data/random/train/m_label'
label_dir='/home/h/Desktop/data/random/train/t_label'
'''
lr = 1e-3
batch_size = 6 # 6이 최대
num_epoch = 150
ckpt_dir = 'random_train copy 2/ckpt'
log_dir = 'random_train copy 2/log'
train_continue = 'on'
## dlalwl
img_dir='/home/h/Desktop/data/random_train (copy)/m_5-34'
label_dir='/home/h/Desktop/data/random_train (copy)t_/5-34'
'''

print("learning rate: %.4e" % lr)
print("batch size: %d" % batch_size)
print("number of epoch: %d" % num_epoch)
print("img_dir : %s" % img_dir)
print("label_dir : %s" % label_dir)
print("ckpt dir: %s" % ckpt_dir)
print("log dir: %s" % log_dir)
print("train_continue: %s" % train_continue)

# make folder if doesn't exist
if not os.path.exists(log_dir):
    os.makedirs(os.path.join(log_dir, 'train'))
    os.makedirs(os.path.join(log_dir, 'val'))
    print('make new logdir')

if not os.path.exists(ckpt_dir):
    os.makedirs(ckpt_dir)
    print('make new ckptdir')
#%%
# data aug & custom dataset
transform=transforms.Compose([transforms.Resize((512, 512)),
                              transforms.RandomHorizontalFlip(), 
                              transforms.RandomVerticalFlip(), 
                              transforms.RandomAffine([-60, 60]),
                              transforms.ColorJitter(brightness=0.5, contrast=0.5),
                              transforms.ToTensor(),
                              transforms.Normalize(mean=0.5, std=0.5)
                              ])
transform_label=transforms.Compose([transforms.Resize((512, 512)),
                              transforms.RandomHorizontalFlip(), 
                              transforms.RandomVerticalFlip(), 
                              transforms.RandomAffine([-60, 60]),
                              transforms.ToTensor(),
                              ])
dataset=CustomDataset(img_dir,label_dir , transform=transform,transform_l= transform_label)

validation_split=.2
shuffle_dataset=True
random_seed=42

dataset_size=len(dataset)# 50
indices=list(range(dataset_size))# [0, 1, ..., 49]
split = int(np.floor(validation_split * dataset_size))# 10(0.2*50)

if shuffle_dataset :# True
    np.random.seed(random_seed)# 난수생성
    np.random.shuffle(indices)# 기존의 배열을 변경->[13, 39, ..., 28, 38]

# train:indices[10], ..., indices[49], val:indices[0], ..., indices[9] 
train_indices, val_indices = indices[split:], indices[:split]
#%%
# Creating PT data samplers and loaders:

# SubsetRandomSampler: 랜덤 리스트 Samples elements randomly from a given list of indices, without replacement.
train_sampler = SubsetRandomSampler(train_indices)
valid_sampler = SubsetRandomSampler(val_indices)

# sampler사용했으니 shuffle=True 불가능 !!
train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, sampler=train_sampler)
validation_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, sampler=valid_sampler)

#%%
# network generate
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
net = UNet().to(device)

# loss function, optimizer

# BCEWithLogitsLoss(): 이진 분류 문제를 풀 때 사용 ->맨마지막아웃풋1
# Sigmoid layer + BCELoss
fn_loss = nn.BCEWithLogitsLoss().to(device)
optim = torch.optim.Adam(net.parameters(), lr=lr)

# variables
num_data_train = len(train_loader)
num_data_val = len(validation_loader)

num_batch_train = np.ceil(num_data_train / batch_size)
num_batch_val = np.ceil(num_data_val / batch_size)
#%%
# functions
# detach(): 기존 Tensor에서 gradient 전파가 안되는 텐서 생성
# 단 storage를 공유하기에 detach로 생성한 Tensor가 변경되면 원본 Tensor도 똑같이 변함
fn_tonumpy = lambda x: x.to('cpu').detach().numpy().transpose(0, 2, 3, 1)
fn_denorm = lambda x, mean, std: (x * std) + mean
fn_class = lambda x: 1.0 * (x > 0.5) # network output image->binary class로 분류
#%%
# set summarywriter to use tensorboard
writer_train = SummaryWriter(log_dir=os.path.join(log_dir, 'train'))
writer_val = SummaryWriter(log_dir=os.path.join(log_dir, 'val'))

# training network
st_epoch=0
#%%# load ckpt if exist
if train_continue == "on":
    net, optim, st_epoch = load(ckpt_dir=ckpt_dir, net=net, optim=optim)

for epoch in range(st_epoch + 1, num_epoch + 1):
    net.train()
    loss_arr = []

    for batch, data in enumerate(train_loader, 1):
        # forward pass
        
        input=data[0].to(device)# torch.Size([4, 3, 512, 512])
        label=data[1].to(device)# torch.Size([4, 1, 512, 512])
        output = net(input)# torch.Size([4, 1, 512, 512])

        # backward pass
        optim.zero_grad()
        loss = fn_loss(output, label)
        # print(loss.item())
        
        loss.backward()
        optim.step()
        
        # loss function
        loss_arr += [loss.item()]
            
        print("TRAIN: EPOCH %04d / %04d | BATCH %04d / %04d | LOSS %.4f" %(epoch, num_epoch, batch, num_batch_train, np.mean(loss_arr)))

        # save to tensorboard
        input = fn_tonumpy(fn_denorm(input, mean=0.5, std=0.5))
        label = fn_tonumpy(label)
        output = fn_tonumpy(fn_class(output))
        
        # NHWC: ([batch, h, w, ch])
        writer_train.add_image('input', input, num_batch_train * (epoch - 1) + batch, dataformats='NHWC')
        writer_train.add_image('label', label, num_batch_train * (epoch - 1) + batch, dataformats='NHWC')
        writer_train.add_image('output', output, num_batch_train * (epoch - 1) + batch, dataformats='NHWC')

    writer_train.add_scalar('loss', np.mean(loss_arr), epoch)
    
    # validation network
    with torch.no_grad():
        net.eval()
        loss_arr = []

        for batch, data in enumerate(validation_loader, 1):
                # forward pass
            input=data[0].to(device)
            label=data[1].to(device)

            output = net(input)

            # 손실함수 계산하기
            loss = fn_loss(output, label)

            loss_arr += [loss.item()]

            print("VALID: EPOCH %04d / %04d | BATCH %04d / %04d | LOSS %.4f" %(epoch, num_epoch, batch, num_batch_val, np.mean(loss_arr)))

            # save to tensorboard
            input = fn_tonumpy(fn_denorm(input, mean=0.5, std=0.5))
            label = fn_tonumpy(label)
            output = fn_tonumpy(fn_class(output))
            
            writer_val.add_image('input', input, num_batch_val * (epoch - 1) + batch, dataformats='NHWC')
            writer_val.add_image('label', label, num_batch_val * (epoch - 1) + batch, dataformats='NHWC')
            writer_val.add_image('output', output, num_batch_val * (epoch - 1) + batch, dataformats='NHWC')
    writer_val.add_scalar('loss', np.mean(loss_arr), epoch)
    
    # save ckpt
    if epoch % 50 == 0:
        save(ckpt_dir=ckpt_dir, net=net, optim=optim, epoch=epoch)

writer_train.close()
writer_val.close()

# %%
