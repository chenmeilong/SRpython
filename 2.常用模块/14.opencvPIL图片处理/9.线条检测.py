import cv2                                     #cv2.HoughLines
import numpy as np

image = cv2.imread("1.jpg")

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

edges = cv2.Canny(gray, 90,110)

lines = cv2.HoughLines(edges,1,np.pi / 180, 250)

print  (lines)                       #只有起点没有 终点

for line in lines:
    rho,theta=line[0]
    a=np.cos(theta)
    b=np.sin(theta)
    x0=a*rho
    y0=b*rho
    x1=int(x0+1000*(-b))
    y1=int(y0+1000*a)
    x2=int(x0-1000*(-b))
    y2=int(y0-1000*a)
    cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 2)


cv2.imshow('HoughLinesP',image)


cv2.waitKey(0)









