#%%
import argparse
import os

import numpy as np
from PIL import Image

#%%
# if __name__ == '__main__':
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--input_path', type=str, help='path of input images')
#     parser.add_argument('--mask_path', type=str, help='path of mask images')
#     parser.add_argument('--result_path', type=str, help='path of label images')
#     args = parser.parse_args()

# input_dir=args.input_path
# mask_dir=args.mask_path
# result_dir=args.result_path

input_dir='/home/h/Desktop/data/random_test/m_label'
mask_dir='random_train/mask'
result_dir='random_train/result'
if not os.path.exists(result_dir):
    os.makedirs(result_dir)
input_lst=os.listdir(input_dir)
mask_lst=[f for f in os.listdir(mask_dir) if os.path.isfile(os.path.join(mask_dir, f))]

input_lst.sort()
mask_lst.sort()

print(len(input_lst))
print(len(mask_lst))

#%%
for i in range(len(input_lst)):
    input_path=os.path.join(input_dir, input_lst[i])
    mask_path=os.path.join(mask_dir, mask_lst[i])
    
    input=Image.open(input_path).resize((512, 512))
    mask=Image.open(mask_path).resize((512, 512))
    
    bg = Image.open('transparence.png').resize((512, 512))
    bg.paste(input,mask)
    bg.save(os.path.join(result_dir,mask_lst[i].replace('t_output','t_label_Unet')))

# %%
