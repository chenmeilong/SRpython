from PyQt5.Qt import *  # 刚开始学习可以这样一下导入
import sys


class Window(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("QFontDialog的学习")
        self.resize(400, 400)
        self.set_ui()

    def set_ui(self):
        font = QFont()
        font.setFamily("宋体")
        font.setPointSize(14)

        fontDialog = QFontDialog(font, self)   #传参
        self.fontDialog = fontDialog

        self.test()
        # fontDialog.show()  #打开对话框的方式
        # fontDialog.open()
        # fontDialog.exec()
        fontDialog.setCurrentFont(font)   #设置当前字体

    def test(self):
        btn = QPushButton(self)
        btn.setText("按钮")
        btn.move(0, 300)

        def font_open_slot():
            print("字体已经被选择好了", self.fontDialog.selectedFont().family())
            print("字体已经被选择好了", self.fontDialog.selectedFont())  #对象

        btn.clicked.connect(lambda: self.fontDialog.open(font_open_slot))


if __name__ == '__main__':
    app = QApplication(sys.argv)

    window = Window()
    window.show()

    sys.exit(app.exec_())