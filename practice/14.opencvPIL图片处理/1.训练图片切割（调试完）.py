# sourceimages     排列操作
import os
import random
from PIL import Image
import pandas  as pd


os.chdir('D:\Wayne\Desktop\银行卡号识别\code\data\sourceimages')
print  (os.listdir())
listdata=os.listdir()
print  (len(listdata))

random.shuffle(listdata)             #随机排列
print(listdata)
print  (len(listdata))

os.chdir('D:\Wayne\Desktop\银行卡号识别\code')

num=0
numlist=[]

for i in listdata:           #就是我们需要  的字符串   文件  名
    im = Image.open('data/sourceimages/'+i)  # 打开图片
    box0 = (0, 0, 30, 46)  ##确定拷贝区域大小       长度  宽度
    box1 = (30, 0, 60, 46)  ##确定拷贝区域大小       长度  宽度
    box2 = (60, 0, 90, 46)  ##确定拷贝区域大小       长度  宽度
    box3 = (90, 0, 120, 46)  ##确定拷贝区域大小       长度  宽度
    region0 = im.crop(box0)  ##将im表示的图片对象拷贝到region中，大小为box
    region1 = im.crop(box1)  ##将im表示的图片对象拷贝到region中，大小为box
    region2 = im.crop(box2)  ##将im表示的图片对象拷贝到region中，大小为box
    region3 = im.crop(box3)  ##将im表示的图片对象拷贝到region中，大小为box

    strdata1=str(num*4)
    strdatanum=i

    region0.save('data/smallimages/'+strdata1+'.png')
    numlist.append(strdatanum[0:1])

    region1.save('data/smallimages/'+str(num*4+1)+'.png')
    numlist.append(strdatanum[1:2])

    region2.save('data/smallimages/'+str(num*4+2)+'.png')
    numlist.append(strdatanum[2:3])

    region3.save('data/smallimages/'+str(num*4+3)+'.png')
    numlist.append(strdatanum[3:4])

    num=num+1

print(numlist)

s = pd.Series(numlist)

print(s)


s.to_csv('test.csv')