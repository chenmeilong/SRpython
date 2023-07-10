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
        btn.clicked.connect(self.test)


        ############################信号copyAvailable###############################

        def textEdit_copyAvailable_slot(arg):
            print("能否复制",arg)
        self.textEdit.copyAvailable.connect(textEdit_copyAvailable_slot)

    def test(self):
        pass

        ############################信号copyAvailable ###############################


if __name__ == '__main__':
    app = QApplication(sys.argv)

    window = Window()
    window.show()

    sys.exit(app.exec_())