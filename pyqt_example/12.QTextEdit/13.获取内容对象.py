
from PyQt5.Qt import *  # 刚开始学习可以这样一下导入
import sys

class Window(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("QTextEdit的学习")
        self.resize(400 ,400)
        self.set_ui()


    def set_ui(self):
        self.textEdit = QTextEdit(self)
        self.textEdit.move(50 ,50)
        self.textEdit.resize(300 ,300)
        self.textEdit.setStyleSheet("background-color:cyan;")
        self.cursor_object()

        btn = QPushButton(self)
        btn.setText("按钮")
        btn.move(0 ,300)
        btn.clicked.connect(lambda :self.cursor_object())

    def setup_conent(self):
        pass

    ############################文本光标对象 --内容和格式的获取 ###############################

    # 注 ： 此时配合按钮一起使用
    def cursor_object(self):
        # QTextCursor   #它的对象方法，编辑器不能很好识别，需要我们自己点进去去找
        QTextCursor
        cursor_obj = self.textEdit.textCursor()

        ############################获取 文本块对象###############################
        QTextBlock
        # print(cursor_obj.block())
        print(cursor_obj.block().text())
        ############################获取 文本块对象###############################

        ############################相应文本块的编号###############################
        print(cursor_obj.blockNumber())
        ############################相应文本块的编号###############################

        ############################当前文本的列表###############################
        print(cursor_obj.currentList())  # 这只是当前没有设置而已

        ############################当前文本的列表###############################

    ############################文本光标对象 --内容和格式的获取   ###############################


if __name__ == '__main__':
    app = QApplication(sys.argv)

    window = Window()
    window.show()

    sys.exit(app.exec_())