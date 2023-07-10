from PyQt5.Qt import * #刚开始学习可以这样一下导入
import sys

class Window(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("QWidget 之事件的api 之显示关闭的学习")
        self.resize(400,400)
        self.set_ui()

        #self.setMouseTracking(True)   #设置鼠标追踪            #按下鼠标移动了>>>>>>不按下移动也能触发事件
        print(self.hasMouseTracking())  # 查看鼠标是否处于跟踪状态
    def set_ui(self):
        pass

    def enterEvent(self, QEvent):
        print("鼠标进入")
    def leaveEvent(self, QEvent):
        print("鼠标离开")

    def mousePressEvent(self, QMouseEvent):
        print("鼠标被按下了")
    def mouseReleaseEvent(self, QMouseEvent):
        print("鼠标被释放了")

    def mouseDoubleClickEvent(self, QMouseEvent):
        print("鼠标被双击了")

    def mouseMoveEvent(self, QMouseEvent):
        print("按下鼠标移动了")
        # print("鼠标移动了",event.globalPos())  # globalPos() 是整个屏
        print("鼠标移动了",QMouseEvent.localPos())  # globalPos() 是控件本身为准

if __name__ == '__main__':
    app =QApplication(sys.argv)

    window = Window()
    window.show()

    sys.exit(app.exec_())