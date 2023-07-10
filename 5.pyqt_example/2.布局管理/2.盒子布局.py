import sys
from PyQt5.QtWidgets import (QWidget, QPushButton,QHBoxLayout, QVBoxLayout, QApplication)
class Example(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
    def initUI(self):
        okButton = QPushButton("OK")
        cancelButton = QPushButton("Cancel")

        hbox = QHBoxLayout()            #盒子布局   水平盒子
        hbox.addStretch(1)            #空白地方平均分配     分配空白区域的比例
        hbox.addWidget(okButton)
        hbox.addStretch(1)
        hbox.addWidget(cancelButton)
        hbox.addStretch(1)

        vbox = QVBoxLayout()      #垂直盒子
        vbox.addStretch(1)
        vbox.addLayout(hbox)

        self.setLayout(vbox)

        self.setGeometry(300, 300, 300, 150)
        self.setWindowTitle('Buttons')
        self.show()


app = QApplication(sys.argv)
ex = Example()
sys.exit(app.exec_())
