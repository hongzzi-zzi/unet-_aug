####unet에 쓸거 !! transform 공부하기
import os
from glob import glob
from torch.utils.data import Dataset

import numpy as np
import torch
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import datasets, transforms
from PIL import Image
import random


class CustomDataset(Dataset):
    def __init__(self, img_dir, label_dir=None, transform=None, transform_l=None):
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.transform = transform
        self.transform_l = transform_l

    def __len__(self):
        return len(os.listdir(self.img_dir))
    
    def __getitem__(self, idx):
        ## pillow 사용
        if self.label_dir!=None:
            img_path=os.path.join(self.img_dir,sorted(os.listdir(self.img_dir))[idx])
            image = Image.open(img_path).convert('RGB')
            
            label_path=os.path.join(self.label_dir,sorted(os.listdir(self.label_dir))[idx])
            label = Image.open(label_path)
            # RGBA 중 마지막
            label=label.split()[-1]
            
            seed=random.randint(1, 10)
            # seed 고정해주기!!!!!!!!!!!!!!
            if self.transform:
                torch.manual_seed(seed)
                image = self.transform(image)
            if self.transform_l:
                torch.manual_seed(seed)
                label = self.transform_l(label)
            return image, label
        else:
            img_path=os.path.join(self.img_dir,sorted(os.listdir(self.img_dir))[idx])
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image, img_path