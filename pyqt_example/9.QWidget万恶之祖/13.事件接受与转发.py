from PyQt5.Qt import * #刚开始学习可以这样一下导入
import sys

class Window(QWidget):
    def mousePressEvent(self, QMouseEvent):
        print("顶层窗口-按下")
class MidWindow(QWidget):
    def mousePressEvent(self, QMouseEvent):
        print("中间窗口-按下")


class Label(QLabel):
    # def mousePressEvent(self, QMouseEvent):
    #     print("标签窗口-按下")
    def mousePressEvent(self, event):
        print("标签窗口-按下")
        print(event.isAccepted())  # 它用来查看事件对象是否被接受了

        # event.accept()  # 这代表的是告诉操作系统，# 我们已经收到了这个事件对象，不需要再次向上转发了

        event.ignore()  #忽略这个事件  事件继续转发
        print(event.isAccepted())

#1,创建app
app  = QApplication(sys.argv)


#2，控件的操作：
#创建控件
window = Window()

mid_window = MidWindow(window)
mid_window.resize(300,300)
mid_window.setAttribute(Qt.WA_StyledBackground,True)  #这样能是下方的qss生效
mid_window.setStyleSheet("background-color:red;")

label = Label(mid_window)
label.setText("我是标签")
label.setStyleSheet("background-color:yellow")
label.move(100,100)

btn = QPushButton(mid_window)
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

# 事件对象的方法accept() 告诉操作系统不让向上转发事件了