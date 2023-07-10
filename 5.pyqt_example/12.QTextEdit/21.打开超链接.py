from PyQt5.Qt import *  # 刚开始学习可以这样一下导入
import sys


class MyTextEdit(QTextEdit):
    def mousePressEvent(self, event):
        QMouseEvent
        # print(event.pos())
        # print(self.anchorAt(event.pos()))  # 这里可以得到 超链接
        url = self.anchorAt(event.pos())
        #下面用拿到的url 打开超链接
        if len(url)>0:
            QDesktopServices.openUrl(QUrl(url))

class Window(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("QTextEdit的学习")
        self.resize(400, 400)
        self.set_ui()

    def set_ui(self):
        self.textEdit = MyTextEdit(self)
        self.textEdit.move(50, 50)
        self.textEdit.resize(300, 300)
        self.textEdit.setStyleSheet("background-color:cyan;")

        btn = QPushButton(self)
        btn.setText("按钮")
        btn.move(0, 300)
        btn.clicked.connect(self.test)

    def test(self):
        ############################tab 设置###############################
        self.textEdit.insertHtml("<a name = 'py123' href = 'http://python123.io'>Python123</a>")
        ############################tab 设置###############################

if __name__ == '__main__':
    app = QApplication(sys.argv)

    window = Window()
    window.show()

    sys.exit(app.exec_())