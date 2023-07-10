from PyQt5.Qt import * #刚开始学习可以这样一下导入
import sys
#1,创建app
app  = QApplication(sys.argv)


#2，控件的操作：
#创建控件
window = QWidget()


#设置控件
window.setWindowTitle("QFrame")
window.resize(500,500)

frame = QFrame(window)
frame.resize(100,100)
frame.move(100,100)
# frame.setStyleSheet("background-color:cyan;")

frame.setFrameShape(QFrame.Box)   #设置外边框
frame.setFrameShadow(QFrame.Raised)  # 设置凸起

frame.setLineWidth(6)  #外线宽
frame.setMidLineWidth(12)  #中线宽   #有的时候是没有中线宽的，例如当形状为Panel 时就没有

frame.setFrameRect(QRect(20,20,60,60))  # 设置框架的矩形


#展示控件
window.show()

#3,进入消息循环
sys.exit(app.exec_())