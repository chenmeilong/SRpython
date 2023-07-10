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
        ############################颜色设置  ###############################
        self.textEdit.setTextBackgroundColor(QColor(200,10,50))
        self.textEdit.setTextColor(QColor(100,200,50))
        ############################颜色设置  ###############################

if __name__ == '__main__':
    app =QApplication(sys.argv)

    window = Window()
    window.show()

    sys.exit(app.exec_())