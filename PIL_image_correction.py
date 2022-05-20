#%%
import os

from PIL import Image, ImageEnhance, ImageOps

%matplotlib inline 
import numpy as np
import matplotlib.pyplot as plt

#%%
path='/home/h/Desktop/data/random_train/m_label'
imglist=os.listdir(path)[:5]
#%%
for i in imglist:
    i_path=os.path.join(path, i)
    print(i_path)
    img=Image.open(i_path)
    i1=ImageEnhance.Sharpness(img).enhance(15)
    i2=ImageEnhance.Color(img).enhance(2)
    i3=ImageEnhance.Contrast(img).enhance(3)
    i4=ImageEnhance.Brightness(img).enhance(1)
    
    ImageEnhance_array = np.hstack((img, i1, i2, i3, i4))
    plt.imshow(ImageEnhance_array)
    plt.show()
#%%
for i in imglist:
    i_path=os.path.join(path, i)
    print(i_path)
    img=Image.open(i_path).convert("RGB") ## mode ??
    i5=ImageOps.grayscale(img).convert("RGB")
    i6=ImageOps.equalize(img)
    i7=ImageOps.invert(img)
    i8=ImageOps.posterize(img, bits=3)
    ImageOps_array = np.hstack((img, i5, i6, i7, i8))
    plt.imshow(ImageOps_array)
    plt.show()
#%%
    
    