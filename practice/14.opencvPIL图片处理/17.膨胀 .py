import cv2          #膨胀的作用是连通边界（白色部分的边界），可以连接在不在一起的物体。
import numpy as np

# 读取名称为 p12.png的图片
img = cv2.imread("data/test_images/1.jpeg",1)

# 转换为黑白图像
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

# 二值化
ret,threshold = cv2.threshold(gray,132,255,cv2.THRESH_BINARY_INV)

# 进行膨胀操作
kernel = np.ones((9,9),np.uint8)
dilate = cv2.dilate(threshold,kernel,iterations=1)

# 显示原图和处理后的图像
cv2.imshow("gray",gray)
cv2.imshow("threshold",threshold)
cv2.imshow("processed",dilate)

cv2.waitKey(0)