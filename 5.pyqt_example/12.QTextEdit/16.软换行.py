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

        btn = QPushButton(self)
        btn.setText("按钮")
        btn.move(0,300)
        btn.clicked.connect(self.test)

    def test(self) :
        ############################软换行 相关###############################
        self.textEdit.setLineWrapMode(QTextEdit.NoWrap)  #不用软换行

        # self.textEdit.setLineWrapMode(QTextEdit.FixedPixelWidth)  #固定像素宽度
        # self.textEdit.setLineWrapColumnOrWidth(100)  # 像素值  或 列数 （根据上面的模式选择）

        self.textEdit.setLineWrapMode(QTextEdit.FixedColumnWidth)  #固定列宽度
        self.textEdit.setLineWrapColumnOrWidth(8)  # 像素值  或 列数 （根据上面的模式选择）
        ############################软换行 相关###############################

if __name__ == '__main__':
    app =QApplication(sys.argv)

    window = Window()
    window.show()

    sys.exit(app.exec_())