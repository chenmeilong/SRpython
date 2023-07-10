from PyQt5.Qt import *  # 刚开始学习可以这样一下导入
import sys


class Window(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("QScrollBar的学习")
        self.resize(400, 400)
        self.set_ui()

    def set_ui(self):
        scrollBar = QScrollBar(self)
        scrollBar.resize(30, 200)
        scrollBar.move(100, 100)

        scrollBar1 = QScrollBar(Qt.Horizontal, self)  # 水平滚动条
        scrollBar1.resize(200, 30)
        scrollBar1.move(200, 300)

        # 信号
        scrollBar.valueChanged.connect(lambda val: print(val))

        # 调整滚动条的长度  通过调整页步长
        scrollBar.setPageStep(50)  # 此时滚动条 约占 1/3

        # 因为有两个滚动条，所以如果用pagedown 操作时，它会不知道操作哪个
        # 此时可以设置捕获键盘
        scrollBar.grabKeyboard()


if __name__ == '__main__':
    app = QApplication(sys.argv)

    window = Window()
    window.show()

    sys.exit(app.exec_())