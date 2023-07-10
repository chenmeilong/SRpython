import cv2
import numpy as np

# 读取名称为 p18.png的图片
img = cv2.imread("data/test_images/12.png",1)
img_org = cv2.imread("data/test_images/12.png",1)

# 得到图片的高和宽
img_height,img_width = img.shape[:2]

# 定义对应的点
points1 = np.float32([[81,30],[378,80],[13,425]])
points2 = np.float32([[0,0],[300,0],[0,400]])

# 变换矩阵M
M = cv2.getAffineTransform(points1,points2)

# 变换后的图像
processed = cv2.warpAffine(img,M,(img_width, img_height))

# 显示原图和处理后的图像
cv2.imshow("org",img_org)
cv2.imshow("processed",processed)

cv2.waitKey(0)