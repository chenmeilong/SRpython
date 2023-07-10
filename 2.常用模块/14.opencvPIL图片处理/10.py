import cv2            #直线检测 cv2.HoughLinesP
import numpy as np  
 
img = cv2.imread("data/test_images/3.jpeg")
 
img = cv2.GaussianBlur(img,(3,3),0)
edges = cv2.Canny(img, 50, 150, apertureSize = 3)
lines = cv2.HoughLines(edges,1,np.pi/180,118)
result = img.copy()
 
#经验参数
minLineLength = 400           #minLineLength参数表示能组成一条直线的最少点的数量，点数量不足的直线将被抛弃。
maxLineGap = 100              #maxLineGap参数表示能被认为在一条直线上的亮点的最大距离。
lines = cv2.HoughLinesP(edges,1,np.pi/180,50,minLineLength,maxLineGap)     #50  参数表示检测一条直线所需最少的曲线交点。

print (lines)
for line in lines:
	x1 = int(round(line[0][0]))
	y1 = int(round(line[0][1]))
	x2 = int(round(line[0][2]))
	y2 = int(round(line[0][3]))
	cv2.line(img,(x1,y1),(x2,y2),(0,255,0),2)


cv2.namedWindow("img",cv2.WINDOW_NORMAL)
cv2.imshow('img', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

