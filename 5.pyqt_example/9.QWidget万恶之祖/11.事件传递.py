#现在鼠标点击触发了事件，但是儿子中没有实现相应的方法，这时这个事件不会被立即丢弃，而是去看它的父亲中有没有实现相应的方法，如果实现就发给父亲，这就是事件的转发。
from PyQt5.Qt import * #刚开始学习可以这样一下导入
import sys

class Window(QWidget):
    def mousePressEvent(self, QMouseEvent):
        print("顶层窗口-按下")
class MidWindow(QWidget):
    def mousePressEvent(self, QMouseEvent):
        print("中间窗口-按下")

class Label(QLabel):
    def mousePressEvent(self, QMouseEvent):
        print("标签控件被按下")

#1,创建app
app  = QApplication(sys.argv)

#2，控件的操作：
#创建控件
window = Window()

mid_window = MidWindow(window)
mid_window.resize(300,300)
mid_window.setAttribute(Qt.WA_StyledBackground,True)  #这样能是下方的qss生效################
mid_window.setStyleSheet("background-color:red;")

label = Label(mid_window)
label.setText("我是标签")
label.setStyleSheet("background-color:yellow")
label.move(100,100)

#设置控件
window.setWindowTitle("事件转发")
window.resize(500,500)

#展示控件
window.show()
#3,进入消息循环
sys.exit(app.exec_())