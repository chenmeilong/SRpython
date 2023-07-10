import cv2           #自适应二值化，用于解决图片中明暗不均导致阈值设置不能有效分割图像。
import numpy as np

img = cv2.imread("data/test_images/9.jpeg")



img=cv2.blur(img,(5,5))
# img=cv2.GaussianBlur(img,(3,3),0)
# img=cv2.medianBlur(img,3)
img=cv2.bilateralFilter(img,9, 41, 41)                 #双边滤波

wigth=img.shape[0]
length=img.shape[1]

gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)   #转为灰度图

# 自适应二值化
threshold = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV,15,3)

kernel = np.ones((2,2),np.uint8)           #腐蚀膨胀
opening = cv2.morphologyEx(threshold,cv2.MORPH_OPEN,kernel)


#经验参数
minLineLength = 400           #minLineLength参数表示能组成一条直线的最少点的数量，点数量不足的直线将被抛弃。
maxLineGap = 200              #maxLineGap参数表示能被认为在一条直线上的亮点的最大距离。
lines = cv2.HoughLinesP(opening,1,np.pi/180,75,minLineLength,maxLineGap)     #75  参数表示检测一条直线所需最少的曲线交点。

for line in lines:
	x1 = int(round(line[0][0]))
	y1 = int(round(line[0][1]))
	x2 = int(round(line[0][2]))
	y2 = int(round(line[0][3]))
	cv2.line(img,(x1,y1),(x2,y2),(0,255,0),2)

allline = np.asarray(lines)
allline=allline.reshape([-1,4])
upline=[]
downline=[]
leftline=[]
rightline=[]
for i in allline:
    anslenth=np.fabs(i[0]-i[2])
    answigth=np.fabs(i[1]-i[3])
    if i[1]<wigth/2 and anslenth>200 and answigth<200 :
        upline.append(i)
        continue
    if i[1]>wigth/2 and anslenth>200 and answigth<200 :
        downline.append(i)
        continue
    if i[0]<length/2 and anslenth<200 and answigth>200 :
        leftline.append(i)
        continue
    if i[0]>length/2 and anslenth<200 and answigth>200 :
        rightline.append(i)
        continue
print (upline)
print ( downline)
print (leftline)
print (rightline)

# kernel = np.ones((3,3),np.uint8)
# closing = cv2.morphologyEx(threshold,cv2.MORPH_CLOSE,kernel)

# cv2.namedWindow("proed",cv2.WINDOW_NORMAL)
# cv2.imshow("proed",threshold )


cv2.namedWindow("pro",cv2.WINDOW_NORMAL)
cv2.imshow("pro",img )

# cv2.namedWindow("processed",cv2.WINDOW_NORMAL)
# cv2.imshow("processed",processed)


cv2.waitKey(0)