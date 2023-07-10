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

    ############################文本光标对象 --文本选中和清空  之设置光标位置###############################

    #注 ： 此时配合按钮一起使用
    def cursor_object(self):
        # QTextCursor   #它的对象方法，编辑器不能很好识别，需要我们自己点进去去找
        QTextCursor
        cursor_obj =self.textEdit.textCursor()

        cursor_obj.setPosition(6,QTextCursor.KeepAnchor)   # 设置锚点不动   本行第六个到最后一个
        # cursor_obj.movePosition(QTextCursor.StartOfLine,1)  #到开头
        # cursor_obj.movePosition(QTextCursor.Up,1)  #到上一行

        # cursor_obj.select(QTextCursor.BlockUnderCursor) #选中一行

        self.textEdit.setTextCursor(cursor_obj)       #需要将文本光标对象设置回去  反向设置
        self.textEdit.setFocus()  #重新获取焦点

        print(cursor_obj.selectedText())
        # print(cursor_obj.selection())
        print(cursor_obj.selection().toPlainText())
        print(cursor_obj.selectedTableCells())  #选中表格的单元格 的使用

        print("选中获取的对应位置为：",cursor_obj.selectionStart(), cursor_obj.selectionEnd())

    ############################文本光标对象 --文本选中和清空   之设置光标位置###############################

if __name__ == '__main__':
    app =QApplication(sys.argv)

    window = Window()
    window.show()

    sys.exit(app.exec_())