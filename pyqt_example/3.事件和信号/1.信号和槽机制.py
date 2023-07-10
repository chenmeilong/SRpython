
import sys
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (QWidget, QLCDNumber, QSlider, QVBoxLayout, QApplication)

class Example(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        lcd = QLCDNumber(self)   #实例化LED数据显示对象
        self.sld = QSlider(Qt.Horizontal)    #滑动条   Qt.Vertical水平滑动条
        print("self.sld",self.sld)
        ##设置最小值
        self.sld.setMinimum(10)
        #设置最大值
        self.sld.setMaximum(50)
        #步长
        self.sld.setSingleStep(3)
        #设置当前值
        self.sld.setValue(20)
        #刻度位置，刻度下方
        self.sld.setTickPosition(QSlider.TicksBelow)
        #设置刻度间距
        self.sld.setTickInterval(5)

        vbox = QVBoxLayout()           #垂直盒子
        print(vbox)
        vbox.addWidget(lcd)
        vbox.addWidget(self.sld)

        self.setLayout(vbox)
        print(lcd.display)
        self.sld.valueChanged.connect(lcd.display)

        self.setGeometry(300, 300, 250, 150)
        self.setWindowTitle('信号和槽机制的')
        self.show()

app = QApplication(sys.argv)
ex = Example()
sys.exit(app.exec_())
