#%%
import os
from PIL import Image
#%%
def lst_resize(lst, directory, w, h):
    for i in lst:
        pth=os.path.join(directory,i)
        img=Image.open(pth)
        if img.size!=(w,h):
            img=img.resize((w,h))
            img.save(pth)
# %%
fi='/home/h/Desktop/data/2/t_label'
lst_i=os.listdir(fi)
lst_i.sort()
print(lst_i)
#%%
lst_resize(lst_i, fi, 512, 512)
#%%
# import os
# from importlib.resources import path
# import shutil
# import random
# from glob import glob
# #%%
# dir_input=glob('/home/h/Desktop/data*/m_label*')
# dir_label=glob('/home/h/Desktop/data*/t_label*')

# lst_input=sorted(os.listdir(dir_input))
# lst_label=sorted(os.listdir(dir_label))

# print(dir_input)
# print(dir_label)
# img_cnts = len(os.listdir(dir_label))