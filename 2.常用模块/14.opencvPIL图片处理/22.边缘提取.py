import cv2
import numpy as np

# 读取名称为 p6.png的图片
img = cv2.imread("data/test_images/8.jpeg",1)

# 高斯模糊
blur = cv2.GaussianBlur(img,(3,3),0)

# Canny提取边缘
processed = cv2.Canny(blur,30,100)

# 显示原图和处理后的图像
cv2.namedWindow("org",cv2.WINDOW_NORMAL)
cv2.imshow("org",img)
cv2.namedWindow("processed",cv2.WINDOW_NORMAL)
cv2.imshow("processed",processed)

cv2.waitKey(0)