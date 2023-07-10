

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
        print("标签窗口-按下")

#1,创建app
app  = QApplication(sys.argv)

#2，控件的操作：
#创建控件
window = Window()

mid_window = MidWindow(window)
mid_window.resize(300,300)
mid_window.setAttribute(Qt.WA_StyledBackground,True)  #这样能是下方的qss生效
mid_window.setStyleSheet("background-color:red;")

label = QLabel(mid_window)  #  注意，这行是QLabel  若是Label则可以触发 print("标签窗口-按下")
label.setText("我是标签")
label.setStyleSheet("background-color:yellow")
label.move(100,100)

btn = QPushButton(mid_window)  #它的父类也是mid_window 应该输出print("中间窗口-按下")    但是按下按钮没有输出则说明 内部是有方法来处理相应的点击的，但是标签QLabel是没有相应的方法来处理的。
#正说明了QPushButton 就是用来被点击的，而QLabel 的天生的才能只是用来展示内容的。
btn.setText("我是按钮")
btn.setStyleSheet("background-color:yellow;")
btn.move(50,50)

#设置控件
window.setWindowTitle("事件转发")
window.resize(500,500)

#展示控件
window.show()

#3,进入消息循环
sys.exit(app.exec_())