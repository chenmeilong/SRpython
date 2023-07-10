
import matplotlib.pyplot as plt # plt 用于显示图片
import matplotlib.image as mpimg # mpimg 用于读取图片
from PIL import Image

import numpy as np


myimg = mpimg.imread('data/cutcard/2.png') # 读取和代码处于同一目录下的图片
plt.imshow(myimg) # 显示显示坐标轴
# plt.show()                  #        图片
plt.axis('off') # 不
a=myimg.shape

hight=a[0]
width=a[1]

print (a[0])         #高
print (a[1])         #长
print (a[2])         # 颜色









