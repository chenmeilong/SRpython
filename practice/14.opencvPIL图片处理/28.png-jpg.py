import os
from PIL import Image

dirname_read="D:\Wayne\Desktop\card\code\data\随机排列需要制作的数据集卡片png/"
dirname_write="D:\Wayne\Desktop\card\code\data\随机排列需要制作的数据集卡片jpg/"
names=os.listdir(dirname_read)
count=0
for name in names:
    img=Image.open(dirname_read+name)
    name=name.split(".")
    if name[-1] == "png":
        name[-1] = "jpg"
        name = str.join(".", name)

        if img.mode=='RGBA':
            r,g,b,a=img.split()
            img=Image.merge("RGB",(r,g,b))
        to_save_path = dirname_write + name
        img.save(to_save_path)
        count+=1
        print(to_save_path, "------conut：",count)
    else:
        continue
