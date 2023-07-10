from PyQt5.Qt import *  # 刚开始学习可以这样一下导入
import sys


class Window(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("QTextEdit的学习")
        self.resize(400, 400)
        self.set_ui()

    def set_ui(self):
        self.textEdit = QTextEdit(self)
        self.textEdit.move(50, 50)
        self.textEdit.resize(300, 300)
        self.textEdit.setStyleSheet("background-color:cyan;")

        btn = QPushButton(self)
        btn.setText("按钮")
        btn.move(0, 300)
        self.setup_conent()
        self.cursor_object()

    def setup_conent(self):
        # self.textEdit.setText("hhhhhhhh")
        # self.textEdit.append("Hello world")
        pass

    ############################文本光标对象方法之插入文本###############################
    def cursor_object(self):
        # QTextCursor   #它的对象方法，编辑器不能很好识别，需要我们自己点进去去找

        cursor_obj = self.textEdit.textCursor()  #获取光标对象
        # 1  插入文本

        # cursor_obj.insertText("I learn Python")

        ############################文本字符格式###############################
        # QTextCharFormat
        textCharFormat = QTextCharFormat()
        textCharFormat.setToolTip("哈哈")    #鼠标停靠提示
        textCharFormat.setFontFamily("隶书")
        textCharFormat.setFontPointSize(16)
        ############################文本字符格式###############################
        cursor_obj.insertText("我要学Python", textCharFormat)

        # 插入html
        cursor_obj.insertHtml("<a href = 'https:python123.io'> Python123</a>")

        # 2

        # 3

    ############################文本光标对象方法之插入文本###############################


if __name__ == '__main__':
    app = QApplication(sys.argv)

    window = Window()
    window.show()

    sys.exit(app.exec_())