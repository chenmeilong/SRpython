from PyQt5.Qt import * #刚开始学习可以这样一下导入
import sys

class Window(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("QTextEdit的学习")
        self.resize(400,400)
        self.set_ui()


    def set_ui(self):
        self.textEdit = QTextEdit(self)
        self.textEdit.move(50,50)
        self.textEdit.resize(300,300)
        self.textEdit.setStyleSheet("background-color:cyan;")

        self.placeHodler()

    ############################占位文本的设置###############################
    def placeHodler(self):
        self.textEdit.setPlaceholderText("请输入您的个人简介")

        print(self.textEdit.placeholderText())

    ############################占位文本的设置###############################

if __name__ == '__main__':
    app =QApplication(sys.argv)

    window = Window()
    window.show()

    sys.exit(app.exec_())