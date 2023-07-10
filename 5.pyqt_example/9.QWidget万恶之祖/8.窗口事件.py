from PyQt5.Qt import * #刚开始学习可以这样一下导入
import sys

class Window(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("QWidget 之事件的api 之显示关闭的学习")
        self.resize(400,400)
        self.set_ui()
    def set_ui(self):
        pass

    def showEvent(self, QShowEvent):   #具体传过来的事件是什么后面说
        print("窗口被展示出来")

    def closeEvent(self, QCloseEvent):  #点×调用它
        print("窗口被关闭了")

    def moveEvent(self, QMoveEvent):
        print("窗口被移动了")

    def resizeEvent(self, QResizeEvent):
        print("窗口改变了尺寸大小")

if __name__ == '__main__':
    app =QApplication(sys.argv)

    window = Window()
    window.show()

    sys.exit(app.exec_())