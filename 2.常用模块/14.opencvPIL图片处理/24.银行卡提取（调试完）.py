import cv2           #自适应二值化，用于解决图片中明暗不均导致阈值设置不能有效分割图像。
import numpy as np

image = cv2.imread("data/test_images/1.jpeg")

img=cv2.blur(image,(5,5))
# img=cv2.GaussianBlur(img,(7,7),0)
# img=cv2.medianBlur(img,7)
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
class Point(object):
    x =0
    y= 0
    def __init__(self, x=0, y=0):
        self.x = x
        self.y = y
class Line(object):
    def __init__(self, p1, p2):
        self.p1 = p1
        self.p2 = p2
def GetLinePara(line):
    line.a =line.p1.y - line.p2.y
    line.b = line.p2.x - line.p1.x
    line.c = line.p1.x *line.p2.y - line.p2.x * line.p1.y
def GetCrossPoint(l1,l2):
    GetLinePara(l1); GetLinePara(l2)
    d = l1.a * l2.b - l2.a * l1.b
    p=Point()
    p.x = (l1.b * l2.c - l2.b * l1.c)*1.0 / d
    p.y = (l1.c * l2.a - l2.c * l1.a)*1.0 / d
    return p
#左上
p1=Point(upline[0][0],upline[0][1])
p2=Point(upline[0][2],upline[0][3])
line1=Line(p1,p2)
p3=Point(leftline[0][0],leftline[0][1])
p4=Point(leftline[0][2],leftline[0][3])
line2=Line(p3,p4)
pointLU = GetCrossPoint(line1,line2)
#右上
p1=Point(upline[0][0],upline[0][1])
p2=Point(upline[0][2],upline[0][3])
line1=Line(p1,p2)
p3=Point(rightline[0][0],rightline[0][1])
p4=Point(rightline[0][2],rightline[0][3])
line2=Line(p3,p4)
pointRU = GetCrossPoint(line1,line2)
#左下
p1=Point(downline[0][0],downline[0][1])
p2=Point(downline[0][2],downline[0][3])
line1=Line(p1,p2)
p3=Point(leftline[0][0],leftline[0][1])
p4=Point(leftline[0][2],leftline[0][3])
line2=Line(p3,p4)
pointLD = GetCrossPoint(line1,line2)
#右下
p1=Point(downline[0][0],downline[0][1])
p2=Point(downline[0][2],downline[0][3])
line1=Line(p1,p2)
p3=Point(rightline[0][0],rightline[0][1])
p4=Point(rightline[0][2],rightline[0][3])
line2=Line(p3,p4)
pointRD = GetCrossPoint(line1,line2)

points1 = np.float32([[pointLU.x,pointLU.y], [pointRU.x,pointRU.y], [pointLD.x,pointLD.y], [pointRD.x,pointRD.y]])    #源图像中待测矩形的四点坐标
points2 = np.float32([[0,0], [793,0], [0,500], [793,500]])      #目标图像中矩形的四点坐标
M = cv2.getPerspectiveTransform(points1, points2)
processed = cv2.warpPerspective(image,M,(793, 500))   #目标图像的shape


cv2.imwrite("data/cutcard/1.png", processed)

cv2.namedWindow("processed",cv2.WINDOW_NORMAL)
cv2.imshow("processed",processed)
cv2.waitKey(0)