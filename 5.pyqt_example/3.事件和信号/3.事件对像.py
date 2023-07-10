import sys
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QWidget, QApplication, QGridLayout, QLabel
class Example(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        grid = QGridLayout()
        grid.setSpacing(10)   #控件上下距离为10
        x = 0
        y = 0
        self.text = "x: {0},  y: {1}".format(x, y)

        self.label = QLabel(self.text, self)
        grid.addWidget(self.label, 0, 0, Qt.AlignTop)  #布局

        self.setMouseTracking(True)     #实现对鼠标的监控
        self.setLayout(grid)

        self.setGeometry(300, 300, 350, 200)
        self.setWindowTitle('Event object')
        self.show()

    def mouseMoveEvent(self, e):
        x = e.x()
        y = e.y()
        text = "x: {0},  y: {1}".format(x, y)
        self.label.setText(text)

app = QApplication(sys.argv)
ex = Example()
sys.exit(app.exec_())
