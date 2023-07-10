#D:\Wayne\Desktop\card\code\data\需要制作的训练数据集卡片
#随机排列银行卡片

import os
import random
from PIL import Image
import cv2 as cv

os.chdir('D:\Wayne\Desktop\card\code\data\需要制作的训练数据集卡片')
print  (os.listdir())
listdata=os.listdir()
print  (len(listdata))

random.shuffle(listdata)             #随机排列
print(listdata)
print  (len(listdata))

os.chdir('D:\Wayne\Desktop\card\code')

num=1

for i in listdata:           #就是我们需要  的字符串   文件  名
    im = Image.open('data/需要制作的训练数据集卡片/'+i)  # 打开图片
    im.save('data/随机排列需要制作的数据集卡片/'+str(num)+'.png')

    num=num+1










