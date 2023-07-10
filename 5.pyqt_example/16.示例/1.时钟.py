from PyQt5 import QtGui, QtCore
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import QApplication,QWidget,QLabel

class AnalogClock(QWidget):
    hourHand = QtGui.QPolygon([
        QtCore.QPoint(10, 8),
        QtCore.QPoint(-10, 8),
        QtCore.QPoint(0, -60)
    ])
    minuteHand = QtGui.QPolygon([
        QtCore.QPoint(8, 8),
        QtCore.QPoint(-8, 8),
        QtCore.QPoint(0, -70)
    ])
    secondHand = QtGui.QPolygon([
        QtCore.QPoint(4, 8),
        QtCore.QPoint(-4, 8),
        QtCore.QPoint(0, -90)
    ])
    hourColor = QtGui.QColor(255, 0, 0)
    minuteColor = QtGui.QColor(0, 255, 0)
    secondColor = QtGui.QColor(0, 0, 255)

    def __init__(self):
        super().__init__()
        timer = QtCore.QTimer(self)
        timer.timeout.connect(self.update)
        timer.start(1000)
        self.setWindowTitle("Analog Clock")
        self.resize(200, 200)

    def paintEvent(self, event):
        side = min(self.width(), self.height()) # 最小边
        time = QtCore.QTime.currentTime()  # 获取系统当前时间

        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)  # 抗锯齿
        painter.translate(self.width() / 2, self.height() / 2) # 坐标位置
        painter.scale(side / 300.0, side / 300.0) # 缩放时 的比例

        painter.setPen(QtGui.QColor(0, 0, 0))
        painter.drawEllipse(-100, -100, 200, 200)  # 画圆。参数是外接矩形左上点和长宽
        # painter.drawEllipse(-10,-10,20,20)

        painter.setPen(AnalogClock.hourColor)
        for i in range(12):  # 整点刻度
            painter.drawLine(88, 0, 96, 0)
            painter.rotate(30.0)
        painter.setPen(AnalogClock.minuteColor)
        for j in range(60):  # 小刻度
            if (j % 5) != 0:
                painter.drawLine(92, 0, 96, 0)
            painter.rotate(6.0)

        painter.setPen(QtCore.Qt.NoPen)
        painter.setBrush(AnalogClock.hourColor)

        painter.save()
        painter.rotate(30.0 * ((time.hour() + time.minute() / 60.0)))
        painter.drawConvexPolygon(AnalogClock.hourHand) # 画三角形
        painter.restore()

        painter.setPen(QtCore.Qt.NoPen)
        painter.setBrush(AnalogClock.minuteColor)


        painter.save()
        painter.rotate(6.0 * (time.minute() + time.second() / 60.0))
        painter.drawConvexPolygon(AnalogClock.minuteHand)
        painter.restore()

        painter.setPen(QtCore.Qt.NoPen)
        painter.setBrush(AnalogClock.secondColor)

        painter.save()
        painter.rotate(6.0 * time.second())
        painter.drawConvexPolygon(AnalogClock.secondHand)
        painter.restore()

        painter.setPen(QtGui.QColor(0, 0, 0))
        painter.drawEllipse(-5, -5, 10, 10)  # 画圆。参数是外接矩形左上点和长宽


if __name__ == '__main__':
    import sys
    app = QApplication(sys.argv)
    # widget = QWidget()
    # widget.setWindowTitle("Hello ZCB")
    # widget.setWindowIcon(QIcon("d:/0.jpg"))
    # widget.show()

    clock = AnalogClock()
    clock.show()

    sys.exit(app.exec())