import cv2         #，闭运算是先膨胀后腐蚀，都可以去除噪声
import numpy as np

# 读取名称为 p13_2.png的图片
img = cv2.imread("data/test_images/1.jpeg",1)

# 转换为黑白图像
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

# 二值化
ret,threshold = cv2.threshold(gray,132,255,cv2.THRESH_BINARY_INV)

# 进行闭运算操作
kernel = np.ones((3,3),np.uint8)
closing = cv2.morphologyEx(threshold,cv2.MORPH_CLOSE,kernel)

# 显示原图和处理后的图像
cv2.imshow("gray",gray)
cv2.imshow("threshold",threshold)
cv2.imshow("processed",closing)

cv2.waitKey(0)