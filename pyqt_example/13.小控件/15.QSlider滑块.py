from PyQt5.Qt import *  # 刚开始学习可以这样一下导入
import sys


class MySlider(QSlider):
    def __init__(self, parent=None, *args, **kwargs):
        super().__init__(parent, *args, **kwargs)
        self.setup_ui()
        self.setMaximum(100)
        self.setMinimum(0)

    def setup_ui(self):
        self.setTickPosition(QSlider.TicksBothSides)   #刻度位置
        self.label = QLabel(self)
        self.label.setText("0")
        self.label.setStyleSheet("background-color:red;")
        self.label.hide()

    def mousePressEvent(self, event):
        QMouseEvent
        super().mousePressEvent(event)
        x = (self.width() - self.label.width()) / 2
        y = (self.maximum() - self.value()) / (self.maximum() - self.minimum()) * (self.height() -self.label.height())
        self.label.show()
        self.label.move(x, y)

    def mouseMoveEvent(self, event):
        super().mouseMoveEvent(event)
        x = (self.width() - self.label.width()) / 2
        y = (self.maximum() - self.value()) / (self.maximum() - self.minimum()) * (self.height() -self.label.height())
        self.label.move(x, y)
        self.label.setText(str(self.value()))
        self.label.adjustSize()    # 它要放在这种检测的事件方法中

    def mouseReleaseEvent(self, event):
        super().mouseReleaseEvent(event)
        self.label.hide()



class Window(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("QSlider 案例的学习")
        self.resize(400, 400)
        self.set_ui()

    def set_ui(self):
        slider = MySlider(self)
        slider.move(200, 200)
        slider.resize(30, 200)


if __name__ == '__main__':
    app = QApplication(sys.argv)

    window = Window()
    window.show()

    sys.exit(app.exec_())