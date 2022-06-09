#%%
import cv2
from IPython.display import clear_output, Image, display
import os
from matplotlib import pyplot as plt
import numpy as np
#%%
path='/home/h/Desktop/data/random/test/m_label'
imglist=os.listdir(path)[:5]
img='/home/h/Desktop/data/random/test/m_label/m_label1_013.png'
#%%
def img_histoequalize(img):
    R, G, B = cv2.split(img)
    output1_R = cv2.equalizeHist(R)
    output1_G = cv2.equalizeHist(G)
    output1_B = cv2.equalizeHist(B)
    final = cv2.merge((output1_R, output1_G, output1_B))
    return final
#%%
def img_whitebalance(img):
    final = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    avg_a = np.average(final[:, :, 1])
    avg_b = np.average(final[:, :, 2])
    for x in range(final.shape[0]):
        for y in range(final.shape[1]):
            l, a, b = final[x, y, :]
			# fix for CV correction
            l *= 100 / 255.0
            final[x, y, 1] = a - ((avg_a - 128) * (l / 100.0) * 1.1)
            final[x, y, 2] = b - ((avg_b - 128) * (l / 100.0) * 1.1)
    final = cv2.cvtColor(final, cv2.COLOR_LAB2BGR)
    return final
#%%
def img_histstreching(img):
    final = cv2.normalize(img, None,0, 255, cv2.NORM_MINMAX)
    return final
#%%
#%%
for i in imglist:
    i_path=os.path.join(path, i)
    print(i_path)
    image=cv2.cvtColor(cv2.imread(i_path),  cv2.COLOR_BGR2RGB)
    i1 = img_histoequalize(image)
    i2=img_whitebalance(image)
    i3=img_histstreching(image)
    final = np.hstack((image, i1, i2, i3))
    plt.imshow(final)
    plt.show()
#%%