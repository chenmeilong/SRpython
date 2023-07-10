import os
import random
import cv2
import numpy as np
import random

import pandas  as pd

os.chdir('D:\Wayne\Desktop\card\code\data\sourceimages')

listdata=os.listdir()
print  (len(listdata))

random.shuffle(listdata)             #随机排列

os.chdir('D:\Wayne\Desktop\card\code')

num=1

for i in listdata:           #就是我们需要  的字符串   文件  名

    img = cv2.imread('data/sourceimages/'+i)
    imgFix = np.zeros((720, 1280, 3), np.uint8)           #创建 全黑  画布

    randpicy=random.randint(23,92)      #随机图片大小宽度
    randpicx=int( randpicy*120/46)      #长度
    imgg = cv2.resize(img,(randpicx, randpicy),cv2.INTER_AREA) #缩小推荐使用 "cv2.INTER_AREA";  扩大推荐使用 “cv2.INTER_CUBIC”
    randx=random.randint(0,(1280-randpicx-1))      #随机位置
    randy=random.randint(0,(720-randpicy-1))
    imgFix[randy:randy+randpicy, randx:randx+randpicx] = imgg#指定位置填充，大小要一样才能填充

    #保存图片
    strnum=str(num)
    straddrnum=strnum.rjust(6,'0')                              #不够前面补起来
    cv2.imwrite("keras_yolo3_master\\VOC2007\\newJPEGImages\\"+straddrnum+".jpg", imgFix)

    strdata="VOC2007/newJPEGImages/"+straddrnum+".jpg"
    for j in range(0,4):
        if i[j:j+1]=='_':
            continue
        boxlabel=[]
        leftupx=randx+ int (randpicx*j/4+1)
        leftupy=randy
        rightdownx=randx+int(randpicx*(j+1)/4+1)
        rightdowny=randy+randpicy
        label= int(i[j:j+1])

        strdata=strdata+' '
        strdata=strdata+str(leftupx)+','
        strdata=strdata+str(leftupy)+','
        strdata=strdata+str(rightdownx)+','
        strdata=strdata+str(rightdowny)+','
        strdata=strdata+str(label)


    f = open("txt",'a',encoding="utf-8")
    f.write(strdata)
    f.write("\n")
    f.close()

    num=num+1

print (num)



