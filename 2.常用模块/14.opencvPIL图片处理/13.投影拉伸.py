import cv2
import numpy as np

# 读取名称为 p19.jpg的图片
img = cv2.imread("data/test_images/11.png",1)
img_org = cv2.imread("data/test_images/11.png",1)

# 得到图片的高和宽
img_height,img_width = img.shape[:2]
print  (img_height,img_width)


# 定义对应的点
points1 = np.float32([[75,55], [340,55], [33,435], [400,433]])    #源图像中待测矩形的四点坐标
points2 = np.float32([[0,0], [500,0], [0,600], [500,600]])      #目标图像中矩形的四点坐标

# 计算得到转换矩阵
M = cv2.getPerspectiveTransform(points1, points2)

print (M)

# 实现透视变换转换
processed = cv2.warpPerspective(img,M,(600, 500))   #目标图像的shape

# 显示原图和处理后的图像
cv2.imshow("org",img_org)
cv2.imshow("processed",processed)

cv2.waitKey(0)