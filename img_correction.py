#%%
import cv2
from IPython.display import clear_output, Image, display
import os
from matplotlib import pyplot as plt
#%%
path='/home/h/Desktop/data/random_train/m_label'
imglist=os.listdir(path)
#%%
def img_contrast(img):
    R, G, B = cv2.split(img)
    output1_R = cv2.equalizeHist(R)
    output1_G = cv2.equalizeHist(G)
    output1_B = cv2.equalizeHist(B)
    final = cv2.merge((output1_R, output1_G, output1_B))
    return final
#%%
for i in imglist:
    i_path=os.path.join(path, i)
    print(i_path)
    img=cv2.imread(i_path)
    # img=cv2.imread('/home/h/Desktop/data/random_train/m_label/m_label4-3_081.png')
    img = img_contrast(img)
    plt.imshow(img)
    plt.show()
# %%
for i in imglist:
    i_path=os.path.join(path, i)
    src = cv2.imread((i_path))

    src_ycrcb = cv2.cvtColor(src, cv2.COLOR_BGR2YCrCb)
    ycrcb_planes = cv2.split(src_ycrcb)


    # 밝기 성분에 대해서만 히스토그램 평활화 수행
    ycrcb_planes[0] = cv2.equalizeHist(ycrcb_planes[0])

    dst_ycrcb = cv2.merge(ycrcb_planes)
    dst = cv2.cvtColor(dst_ycrcb, cv2.COLOR_YCrCb2BGR)

    plt.imshow(dst)
    plt.show()

# %%
