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

        self.cursor_object()
        btn = QPushButton(self)
        btn.setText("按钮")
        btn.move(0, 300)
        btn.clicked.connect(lambda: self.cursor_object())

    def setup_conent(self):
        pass

    ############################文本光标对象 --清空和判定###############################

    # 注 ： 此时配合按钮一起使用
    def cursor_object(self):
        # QTextCursor   #它的对象方法，编辑器不能很好识别，需要我们自己点进去去找
        QTextCursor
        cursor_obj = self.textEdit.textCursor()

        # cursor_obj.clearSelection()  # 取消文本选中  它需要反向设置
        # self.textEdit.setTextCursor(cursor_obj)

        # cursor_obj.removeSelectedText()  #移除选中文本


        #向后删   如果是选中文本的话删除选中文本
        # cursor_obj.deleteChar()
        #向前删   如果是选中文本的话删除选中文本
        cursor_obj.deletePreviousChar()

        self.textEdit.setFocus()

    ############################文本光标对象 --清空和判定###############################


if __name__ == '__main__':
    app = QApplication(sys.argv)

    window = Window()
    window.show()

    sys.exit(app.exec_())