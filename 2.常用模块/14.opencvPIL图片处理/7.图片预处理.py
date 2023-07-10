import matplotlib.pyplot as plt # plt 用于显示图片
import matplotlib.image as mpimg # mpimg 用于读取图片
import numpy as np
import tensorflow as tf
import  pandas as pd

myimg1 = mpimg.imread('smallimages/0.png') # 读取和代码处于同一目录下的图片
# full1 = np.reshape(myimg, [1, 46, 30, 3])


myimg2= mpimg.imread('smallimages/1.png') # 读取和代码处于同一目录下的图片

# full2 = np.reshape(myimg, [1, 46, 30, 3])

myimg3=np.stack((myimg1,myimg2),0)


print (myimg1.shape)
print (myimg3.shape)



# plt.imshow(myimg) # 显示图片
# plt.axis('off') # 不显示坐标轴
# plt.show()
# print(myimg.shape)
#





