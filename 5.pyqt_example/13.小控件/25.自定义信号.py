from PyQt5.Qt import * #刚开始学习可以这样一下导入
import sys

class Btn(QPushButton):
    right_clicked = pyqtSignal(str)

    def mousePressEvent(self,event):
        super().mousePressEvent(event)

        if event.button() == Qt.RightButton:  #这里解决了什么时候发射信号
            # print("应该发射右击信号")
            self.right_clicked.emit("传递的字符串参数")   #这里解决了如何将自定义信号发射出去

class Window(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("信号的学习")
        self.resize(400,400)
        self.set_ui()
    def set_ui(self):
        btn = Btn("按钮",self)
        btn.right_clicked.connect(self.right_clicked_slot)

    def right_clicked_slot(self,msg):
        print(msg)
        print("右键被点击了")


if __name__ == '__main__':
    app =QApplication(sys.argv)

    window = Window()
    window.show()

    sys.exit(app.exec_())