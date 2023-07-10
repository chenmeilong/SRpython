
import sys
from PyQt5.QtWidgets import QWidget, QApplication
from PyQt5.QtGui import QPainter, QColor, QFont
from PyQt5.QtCore import Qt

class Example(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
    def initUI(self):
        self.text = "涂鸦要涂的有灵魂"
        self.setGeometry(300, 300, 280, 170)
        self.setWindowTitle('绘画板')
        self.show()

    def paintEvent(self, event):
        qp = QPainter()
        qp.begin(self)
        self.drawText(event, qp,168, 34, 243)
        qp.end()
        # qp1 = QPainter()
        # qp1.begin(self)
        # self.drawText(event, qp1,168, 34, 23)
        # qp1.end()

    def drawText(self, event, qp, r,g,b):
        qp.setPen(QColor(r,g,b))
        qp.setFont(QFont('微软雅黑', 15))
        qp.drawText(event.rect(), Qt.AlignCenter, self.text)


if __name__ == '__main__':

    app = QApplication(sys.argv)
    ex = Example()
    sys.exit(app.exec_())
