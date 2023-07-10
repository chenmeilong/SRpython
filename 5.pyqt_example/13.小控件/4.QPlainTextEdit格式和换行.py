from PyQt5.Qt import *  # 刚开始学习可以这样一下导入
import sys


class Window(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("QPlainTextEdit的学习")
        self.resize(500, 500)
        self.set_ui()

    def set_ui(self):
        self.plainTextEdit = QPlainTextEdit(self)
        self.plainTextEdit.resize(300, 300)
        self.plainTextEdit.move(100, 100)
        self.test()

    def test(self):
        ############################格式设置###############################
        textCharFormat = QTextCharFormat()

        ############################QtextCharFormat 的设置###############################
        textCharFormat.setFontUnderline(True)
        textCharFormat.setUnderlineColor(QColor(20, 200, 200))

        ############################QtextCharFormat 的设置###############################
        self.plainTextEdit.setCurrentCharFormat(textCharFormat)
        ############################格式设置###############################


        ############################自动换行###############################
        print(self.plainTextEdit.lineWrapMode())  #默认是软换行
        self.plainTextEdit.setLineWrapMode(0)   # 改变它  变成不是自动换行
        ############################自动换行###############################




if __name__ == '__main__':
    app = QApplication(sys.argv)

    window = Window()
    window.show()

    sys.exit(app.exec_())