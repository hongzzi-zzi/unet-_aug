#%%
import os
from importlib.resources import path
import shutil
import random
from glob import glob
from PIL import Image, ImageEnhance, ImageOps

#%%
dir_input=glob('/home/h/Desktop/data/*/m_label/*')
dir_label=glob('/home/h/Desktop/data/*/t_label/*')
print(len(dir_input))
print(len(dir_label))
lst_input=sorted(dir_input)
lst_label=sorted(dir_label)
print(lst_input[50])
print(lst_label[50])
lst_all=[[i, l]for i, l in zip(lst_input, lst_label)]
img_cnt=len(lst_all)
#%%
test_cnt=int(img_cnt*0.1)
train_cnt=img_cnt-test_cnt
random.shuffle(lst_all)
lst_test=lst_all[:test_cnt]
print(lst_test)
lst_train=lst_all[test_cnt:]
print(lst_train)
print(len(lst_test))
print(len(lst_train))

#%%
train_path="/home/h/Desktop/data/random2/train"
test_path="/home/h/Desktop/data/random2/test"

if os.path.exists(test_path):
    shutil.rmtree(test_path)

if os.path.exists(train_path):
    shutil.rmtree(train_path)
#%%
os.makedirs(os.path.join(test_path, 'm_label'))
os.makedirs(os.path.join(test_path, 't_label'))
os.makedirs(os.path.join(train_path, 'm_label'))
os.makedirs(os.path.join(train_path, 't_label'))
#%%
for i in lst_test:
    shutil.copyfile(i[0], os.path.join(test_path, i[0].split('/')[-2], i[0].split('/')[-1]))
    shutil.copyfile(i[1], os.path.join(test_path, i[1].split('/')[-2], i[1].split('/')[-1]))
for i in lst_train:
    shutil.copyfile(i[0], os.path.join(train_path, i[0].split('/')[-2], i[0].split('/')[-1]))
    shutil.copyfile(i[1], os.path.join(train_path, i[1].split('/')[-2], i[1].split('/')[-1]))
#%%
# a='/home/h/Desktop/data/random2/train/m_label'
# b='/home/h/Desktop/data/random2/train/t_label'
# lsta=sorted(os.listdir(a))
# lstb=sorted(os.listdir(b))
# for i in range(len(lsta)):
#     if lsta[i].split('_')[-1]!=lstb[i].split('_')[-1]:
#         print(lsta[i])
#         print(lstb[i])
#         print(i)
# %%
