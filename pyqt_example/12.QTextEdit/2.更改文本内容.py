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

        self.setup_conent()


    ############################文本内容的设置###############################
    def setup_conent(self):
        # 设置普通文本
        # self.textEdit.setPlainText("<h1>Life is short,")  #第一次设置完之后，光标在最前面
        # self.textEdit.insertPlainText("I learn Python!</h1>")
        # print(self.textEdit.toHtml())  #  输出的是html的框架

        # 设置富文本
        # self.textEdit.setHtml("<h1>Life is short,")   #第一次设置完之后，光标在最前面
        # self.textEdit.insertHtml("I learn Python!</h1>")
        # print(self.textEdit.toPlainText())

        # 自动识别
        # self.textEdit.setText("<h1>Life is short,I learn Python!</h1>")

        self.textEdit.setText("hhhhhhhh")
        self.textEdit.append("Hello world")  # 但是此时光标的位置仍然是开始


        # self.textEdit.setText("") #清空内容
        self.textEdit.clear()  # 它也可以

    ############################文本内容的设置###############################


if __name__ == '__main__':
    app = QApplication(sys.argv)

    window = Window()
    window.show()

    sys.exit(app.exec_())