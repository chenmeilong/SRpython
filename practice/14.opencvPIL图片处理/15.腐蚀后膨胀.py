import cv2            #开运算是先腐蚀后膨胀
import numpy as np

# 读取名称为 p13.png的图片
img = cv2.imread("data/test_images/1.jpeg")

# 转换为黑白图像
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

# 二值化
ret,threshold = cv2.threshold(gray,132,255,cv2.THRESH_BINARY_INV)

# 进行开运算操作
kernel = np.ones((3,3),np.uint8)
opening = cv2.morphologyEx(threshold,cv2.MORPH_OPEN,kernel)

# 显示原图和处理后的图像
cv2.imshow("gray",gray)
cv2.imshow("threshold",threshold)
cv2.imshow("processed",opening)

cv2.waitKey(0)