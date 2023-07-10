import os
import random
from PIL import Image
from PIL import ImageFilter                         ## 调取ImageFilter
import pandas  as pd


im = Image.open('data/test_images/1.jpeg')  # 打开图片
print(im.mode) ## 打印出模式信息
print(im.size) ## 打印出尺寸信息

new_im = im.convert('L')
print(new_im.mode)
new_im.show()
im.show()


# 滤波器
# imgF = Image.open('data/test_images/1.jpeg')  # 打开图片
# bluF = imgF.filter(ImageFilter.BLUR) ##均值滤波
# conF = imgF.filter(ImageFilter.CONTOUR) ##找轮廓
# edgeF = imgF.filter(ImageFilter.FIND_EDGES) ##边缘检测
# imgF.show()
# bluF.show()
# conF.show()
# edgeF.show()













