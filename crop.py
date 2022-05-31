#%%
from PIL import Image
import os
path='/home/h/Desktop/data/2/m_label'
lst=os.listdir(path)
for i in lst:
    image1 = Image.open(os.path.join(path,i))
    # image1.show()
    
    #이미지의 크기 출력
    print(image1.size)
    #%%
    # 이미지 자르기 crop함수 이용 ex. crop(left,up, rigth, down)
    croppedImage=image1.crop((0,100,512,512))
    
    # croppedImage.show()
    
    print("잘려진 사진 크기 :",croppedImage.size)
    
    croppedImage.save(os.path.join(path,i))
# %%
