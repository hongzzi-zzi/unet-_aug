#%%
import os

from PIL import Image, ImageEnhance
from PIL import ImageOps

%matplotlib inline 
import numpy as np
import matplotlib.pyplot as plt

#%%
path='/home/h/Desktop/data/random_train/m_label'
imglist=os.listdir(path)[:5]
#%%
def show_image(flist):
    final = np.hstack((flist))
    plt.show(flist)
    
#%%
for i in imglist:
    i_path=os.path.join(path, i)
    print(i_path)
    image=Image.open(i_path)
    i1=ImageEnhance.Sharpness(image).enhance(10)
    i2=ImageEnhance.Color(image).enhance(2)
    i3=ImageEnhance.Contrast(image).enhance(3)
    i4=ImageEnhance.Brightness(image).enhance(3)
    ImageEnhance = np.hstack((image, i1, i2, i3, i4))
    plt.imshow(ImageEnhance)
    plt.show()
    i5=ImageOps.grayscale(image)
    i6=ImageOps.equalize(image)
    i7=ImageOps.invert(image)
    i8=ImageOps.mirror(image)
    i9=ImageOps.flip(image)
    i10=ImageOps.posterize(image)
    ImageOps = np.hstack((image, i5, i6, i7, i8, i9, i10))
    plt.imshow(ImageOps)
    plt.show()
#%%
    
    