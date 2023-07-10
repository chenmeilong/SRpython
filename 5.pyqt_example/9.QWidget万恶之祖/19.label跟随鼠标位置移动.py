from PyQt5.Qt import * #刚开始学习可以这样一下导入
import sys


class Window(QWidget):
    def __init__(self):
        super().__init__()
        #设置控件
        self.setWindowTitle("鼠标相关的案例")
        self.resize(500,500)
        self.setMouseTracking(True)
        self.setMyCursor()
        self.setLabel()

    def setMyCursor(self):
        pixmap = QPixmap("icon.png").scaled(50,50)
        cursor = QCursor(pixmap)
        self.setCursor(cursor)
    def setLabel(self):
        self.label  = QLabel(self)
        self.label.setText("Life is short,I learn Python!")
        self.label.move(100,100)
        self.label.setStyleSheet("background-color:cyan;")

    def mouseMoveEvent(self, event):
        print("鼠标移动",event.localPos())
        self.label.move(event.localPos().x(),event.localPos().y())

#1,创建app
app  = QApplication(sys.argv)
window = Window()
#展示控件
window.show()
#3,进入消息循环
sys.exit(app.exec_())