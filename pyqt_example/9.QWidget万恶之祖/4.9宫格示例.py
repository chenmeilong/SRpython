import math
import sys
from PyQt5.QtWidgets import QWidget, QLabel, QApplication, QPushButton

class Window(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("QT的学习")
        self.resize(500,500)
        self.num = 9
        self.col = 3
        self.set_ui()

    def set_ui(self):
        temp = 0
        width = self.width()/self.col
        height = self.height()/math.ceil(self.num / self.col)
        for rIdx in range(math.ceil(self.num / self.col)):
            for cIdx in range(self.col):
                temp += 1
                if temp >self.num:
                    break
                w = QWidget(self)
                w.resize(width, height)
                w.move(cIdx*width,rIdx*height)
                w.setStyleSheet("background-color:red;border:1px solid yellow;")
if __name__ == '__main__':
    app =QApplication(sys.argv)
    window = Window()
    window.show()
    sys.exit(app.exec_())