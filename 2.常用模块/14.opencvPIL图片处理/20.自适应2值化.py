import cv2           #自适应二值化，用于解决图片中明暗不均导致阈值设置不能有效分割图像。
import numpy as np

# 读取名称为 p8.png的图片
img = cv2.imread("data/test_images/9.jpeg",0)

# 自适应二值化
processed = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C, \
                cv2.THRESH_BINARY,15,2)

# 显示原图和处理后的图像
# cv2.namedWindow("org",cv2.WINDOW_NORMAL)
# cv2.imshow("org",img)
cv2.namedWindow("processed",cv2.WINDOW_NORMAL)
cv2.imshow("processed",processed)

cv2.waitKey(0)