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

    def setup_conent(self):
        self.textEdit.setText("hhhhhhhh")
        self.textEdit.append("Hello world")  # 但是此时光标的位置仍然是开始

        ############################文本文档对象###############################
        print(self.textEdit.document()) # <PyQt5.QtGui.QTextDocument object at 0x000002A45A01AD38>

        print(self.textEdit.textCursor()) #光标对象 # <PyQt5.QtGui.QTextCursor object at 0x000001C8CB85A358>


############################文本文档对象###############################


if __name__ == '__main__':
    app = QApplication(sys.argv)

    window = Window()
    window.show()

    sys.exit(app.exec_())