
#D:\Wayne\Desktop\card\code\keras_yolo3_master\test\data
import cv2
import os

cap = cv2.VideoCapture(0)
cap.set(3,1280) #设置分辨率
cap.set(4,720)

while (1):
    ret, frame = cap.read()
    print (ret)
    k = cv2.waitKey(1)              #等待  按键  按下
    if k == 27:                     #是否按下 esc  按键
        break
    elif k == ord('s'):             #按下  s键拍照
        os.chdir('D:\\Wayne\\Desktop\\card\code\\keras_yolo3_master\\test\\data')
        listdata = os.listdir()
        namelist = []
        for i in listdata:
            namelist.append(int(i[0:-4]))
        namelist.sort()  # 修改原列表的排序
        print ("save as",namelist[-1]+1,".jpg")
        cv2.imwrite('D:\\Wayne\\Desktop\\card\code\\keras_yolo3_master\\test\\data\\' + str(namelist[-1]+1) + '.jpg', frame)
    h_flip = cv2.flip(frame, 1)              #图像左右翻转
    cv2.imshow("capture",h_flip)
cap.release()
cv2.destroyAllWindows()

