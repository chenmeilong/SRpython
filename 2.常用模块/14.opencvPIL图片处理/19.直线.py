import cv2
import numpy as np

# 读取名称为 p9.png的图片
org = cv2.imread("data/test_images/1.jpeg",1)
img = cv2.imread("data/test_images/1.jpeg",1)
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# 提取边缘
edges = cv2.Canny(gray,30,250,apertureSize=3)
# 提取直线
lines = cv2.HoughLines(edges,1,np.pi/180,200)

for line in lines:
    for rho,theta in line:
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a*rho
        y0 = b*rho
        x1 = int(x0 + 1000*(-b))
        y1 = int(y0 + 1000*(a))
        x2 = int(x0 - 1000*(-b))
        y2 = int(y0 - 1000*(a))
        #把直线显示在图片上
        cv2.line(img,(x1,y1),(x2,y2),(0,0,255),2)

# 显示原图和处理后的图像
cv2.imshow("org",org)
cv2.imshow("processed",img)

cv2.waitKey(0)