
from PyQt5.QtWidgets import (QWidget, QSlider,QLabel, QApplication)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap
import sys

class Example(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        sld = QSlider(Qt.Horizontal, self)
        sld.setFocusPolicy(Qt.NoFocus)
        sld.setGeometry(30, 40, 100, 30)
        sld.valueChanged[int].connect(self.changeValue)

        self.label = QLabel(self)
        self.label.setPixmap(QPixmap('images/logo.png'))
        self.label.setGeometry(160, 40, 100, 80)        #位置和大小

        self.setGeometry(300, 300, 280, 170)
        self.setWindowTitle('这是标题')
        self.show()

    def changeValue(self, value):
        print(value)
        if value == 0:
            self.label.setPixmap(QPixmap('images/1.png'))
        elif value > 0 and value <= 30:
            self.label.setPixmap(QPixmap('images/2.png'))
        elif value > 30 and value < 80:
            self.label.setPixmap(QPixmap('images/3.png'))
        else:
            self.label.setPixmap(QPixmap('images/4.png'))

app = QApplication(sys.argv)
ex = Example()
sys.exit(app.exec_())

