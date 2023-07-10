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
        self.cursor_object()

        btn = QPushButton(self)
        btn.setText("按钮")
        btn.move(0,300)
        btn.clicked.connect(lambda :self.cursor_object())




    def setup_conent(self) :
        pass

    ############################文本光标对象 --格式设置和合并方法 之合并当前字符格式###############################

    #注 ： 此时配合按钮一起使用
    def cursor_object(self):
        # QTextCursor   #它的对象方法，编辑器不能很好识别，需要我们自己点进去去找
        QTextCursor
        cursor_obj =self.textEdit.textCursor()


        textCharFormat1 = QTextCharFormat()
        textCharFormat2 = QTextCharFormat()
        ############################QTextCharFormat 的设置###############################
        textCharFormat1.setFontFamily("幼圆")
        textCharFormat1.setFontUnderline(True)
        textCharFormat1.setFontOverline(True)

        textCharFormat2.setFontStrikeOut(True)

        ############################QTextCharFormat 的设置###############################
        cursor_obj.setCharFormat(textCharFormat1)
        # cursor_obj.setCharFormat(textCharFormat2)  # 它会量1覆盖掉的

        cursor_obj.mergeCharFormat(textCharFormat2)  # 它会将 1 和 2 合并
    ############################文本光标对象 --格式设置和合并方法   之合并当前字符格式###############################



if __name__ == '__main__':
    app =QApplication(sys.argv)

    window = Window()
    window.show()

    sys.exit(app.exec_())