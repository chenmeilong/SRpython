import cv2
import numpy as np

img = cv2.imread("data/test_images/1.jpeg")
print (img.shape)     #形状
print (img.size)       #像素大小
print (img.dtype)      #类型

imgg = cv2.resize(img,(480,360),cv2.INTER_AREA)  #缩小推荐使用 "cv2.INTER_AREA";  扩大推荐使用 “cv2.INTER_CUBIC”
cv2.imshow("imgg", imgg)      #缩小或者扩大
cv2.waitKey()


# cv2.imwrite("data/cutcard/1.png", processed)