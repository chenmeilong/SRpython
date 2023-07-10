from PyQt5.Qt import * #刚开始学习可以这样一下导入
import sys

class Window(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("QWidget 之事件的api 之显示关闭的学习")
        self.resize(400,400)
        self.set_ui()
        self.setMouseTracking(True)   #设置鼠标追踪

    def set_ui(self):
        pass

    def keyPressEvent(self, QKeyEvent):
        print("键盘上某个键被按下了")

    def keyReleaseEvent(self, QKeyEvent):
        print("键盘上某个键被释放了")

if __name__ == '__main__':
    app =QApplication(sys.argv)
    window = Window()
    window.show()
    sys.exit(app.exec_())

#