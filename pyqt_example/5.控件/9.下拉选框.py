
from PyQt5.QtWidgets import (QWidget, QLabel,
    QComboBox, QApplication)
import sys

class Example(QWidget):

    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.lbl = QLabel("Ubuntu", self)
        combo = QComboBox(self)         #下拉框
        combo.addItem("Ubuntu")
        combo.addItem("Windows")
        combo.addItem("centos")
        combo.addItem("deepin")
        combo.addItem("redhat")
        combo.addItem("debain")
        combo.move(50, 50)
        self.lbl.move(50, 150)

        combo.activated[str].connect(self.onActivated)       #下拉框响应事件

        self.setGeometry(300, 300, 300, 200)      #设置屏幕位置和长宽
        self.setWindowTitle('下拉选框练习 ')
        self.show()

    def onActivated(self, text):
        self.lbl.setText(text)
        self.lbl.adjustSize()

app = QApplication(sys.argv)
ex = Example()
sys.exit(app.exec_())
