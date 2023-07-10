
import cv2
import numpy as np

# 读取名称为 p11.png的图片
img = cv2.imread("data/test_images/1.jpeg",1)

# 转换为黑白图像
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

# 二值化
ret,threshold = cv2.threshold(gray,70,255,cv2.THRESH_BINARY)

# 进行腐蚀操作
kernel = np.ones((5,5),np.uint8)
erosion = cv2.erode(threshold,kernel,iterations=1)

# 显示原图和处理后的图像
cv2.imshow("gray",gray)
cv2.imshow("threshold",threshold)
cv2.imshow("processed",erosion)

cv2.waitKey(0)





