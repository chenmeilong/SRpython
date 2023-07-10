
from PyQt5.QtWidgets import QPushButton, QWidget, QApplication
from PyQt5.QtCore import Qt, QMimeData
from PyQt5.QtGui import QDrag
import sys

class Button(QPushButton):
    def __init__(self, title, parent):
        super().__init__(title, parent)

    def mouseMoveEvent(self, e):
        if e.buttons() != Qt.RightButton:
            return
        mimeData = QMimeData()
        drag = QDrag(self)
        drag.setMimeData(mimeData)
        drag.setHotSpot(e.pos() - self.rect().topLeft())

        dropAction = drag.exec_(Qt.MoveAction)

    def mousePressEvent(self, e):
        super().mousePressEvent(e)
        if e.button() == Qt.LeftButton:
            print('按我嘎哈')


class Example(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setAcceptDrops(True)
        self.button = Button('来吧！！！', self)
        self.button.move(100, 65)
        self.setWindowTitle('点击还能挪')
        self.setGeometry(300, 300, 280, 150)

    def dragEnterEvent(self, e):
        e.accept()


    def dropEvent(self, e):
        position = e.pos()
        self.button.move(position)
        e.setDropAction(Qt.MoveAction)
        e.accept()


if __name__ == '__main__':

    app = QApplication(sys.argv)
    ex = Example()
    ex.show()
    app.exec_()
"""
拖拽
在GUI里，拖放是指用户点击一个虚拟的对象，拖动，然后放置到另外一个对象上面的动作。
一般情况下，需要调用很多动作和方法，创建很多变量。
拖放能让用户很直观的操作很复杂的逻辑。
一般情况下，我们可以拖放两种东西：数据和图形界面。
把一个图像从一个应用拖放到另外一个应用上的实质是操作二进制数据。
把一个表格从Firefox上拖放到另外一个位置 的实质是操作一个图形组。
"""
