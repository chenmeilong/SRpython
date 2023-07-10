import cv2
import numpy as np

# 读取名称为 p7.png的图片
img = cv2.imread("data/test_images/1.jpeg",0)

# 二值化
ret,processed = cv2.threshold(img,175,255,cv2.THRESH_BINARY)   #调节175的阈值

# 显示原图和处理后的图像
cv2.namedWindow("org",cv2.WINDOW_NORMAL)
cv2.imshow("org",img)
cv2.namedWindow("processed",cv2.WINDOW_NORMAL)
cv2.imshow("processed",processed)

cv2.waitKey(0)