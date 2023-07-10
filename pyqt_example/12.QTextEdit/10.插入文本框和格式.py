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
        self.setup_conent()
        self.cursor_object()

    def setup_conent(self) :
        # self.textEdit.setText("hhhhhhhh")
        # self.textEdit.append("Hello world")
        pass

    ############################文本光标对象方法之文本块###############################
    def cursor_object(self):
        # QTextCursor   #它的对象方法，编辑器不能很好识别，需要我们自己点进去去找
        QTextCursor
        cursor_obj =self.textEdit.textCursor()

        # 在设置QTextBlockFormat 段落级别的时候也可以设置字符级别的格式
        textBlockFormat = QTextBlockFormat()
        textCharFormat = QTextCharFormat()
        ############################QTextBlockFormat 和 QTextCharFormat的设置###############################

        #对段落的设置
        textBlockFormat.setAlignment(Qt.AlignRight)  #右对齐
        textBlockFormat.setRightMargin(100)  #右边距是 100
        textBlockFormat.setIndent(3)  #缩进是3个 tab

        #对字符的设置
        textCharFormat.setFontFamily("隶书")
        textCharFormat.setFontPointSize(20)
        textCharFormat.setFontItalic(True)

        ############################QTextBlockFormat 和 QTextCharFormat的设置###############################
        cursor_obj.insertBlock(textBlockFormat,textCharFormat)

    ############################文本光标对象方法之文本块###############################

if __name__ == '__main__':
    app =QApplication(sys.argv)
    window = Window()
    window.show()
    sys.exit(app.exec_())