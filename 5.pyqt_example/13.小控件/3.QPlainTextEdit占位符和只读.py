
#QPlainTextEdit和QTextEdit差不多（但是它不是继承QTextEdit），但是它更适合大的文本！

from PyQt5.Qt import * #刚开始学习可以这样一下导入
import sys

class Window(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("QPlainTextEdit的学习")
        self.resize(500,500)
        self.set_ui()
    def set_ui(self):
        self.plainTextEdit = QPlainTextEdit(self)     #按行滚动
        self.plainTextEdit.resize(300,300)
        self.plainTextEdit.move(100,100)
        self.test()

    def test(self):
        ############################占位提示文本###############################
        self.plainTextEdit.setPlaceholderText("请输入你的个人信息")
        print(self.plainTextEdit.placeholderText())
        ############################占位提示文本###############################

        ############################只读设置###############################
        self.plainTextEdit.setReadOnly(True)
        print(self.plainTextEdit.isReadOnly())
        ############################只读设置###############################



if __name__ == '__main__':
    app =QApplication(sys.argv)

    window = Window()
    window.show()

    sys.exit(app.exec_())