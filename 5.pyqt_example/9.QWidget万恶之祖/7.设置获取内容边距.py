from PyQt5.Qt import * #刚开始学习可以这样一下导入
import sys
#1,创建app
app  = QApplication(sys.argv)


#2，控件的操作：
#创建控件
window = QWidget()


#设置控件
window.setWindowTitle("内容边距的设定")
window.resize(500,500)


label = QLabel(window)
label.setText("Life is short,I learn Python!")
label.resize(300,300)
label.setStyleSheet("background-color:cyan;border:1px solid red")

label.setContentsMargins(100,0,0,0)  # 设置内容边距，四个参数是边距顺时针，从左开始
print(label.contentsRect())
print(label.getContentsMargins())  #打印各个边内容边距设置的具体值

#展示控件
window.show()


#3,进入消息循环
sys.exit(app.exec_())
'''
    输出：
    PyQt5.QtCore.QRect(100, 0, 200, 300)
'''