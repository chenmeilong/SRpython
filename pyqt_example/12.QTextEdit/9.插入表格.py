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

        btn = QPushButton(self)
        btn.setText("按钮")
        btn.move(0, 300)
        self.setup_conent()
        self.cursor_object()

    def setup_conent(self):
        # self.textEdit.setText("hhhhhhhh")
        # self.textEdit.append("Hello world")
        pass

    ############################文本光标对象方法之插入表格###############################
    def cursor_object(self):
        # QTextCursor   #它的对象方法，编辑器不能很好识别，需要我们自己点进去去找
        QTextCursor
        cursor_obj = self.textEdit.textCursor()

        # cursor_obj.insertTable(5,3)  #插入5行  3 列

        QTextTableFormat
        textTableFormat = QTextTableFormat()

        ############################QTextTableFormat 的设置###############################
        textTableFormat.setAlignment(Qt.AlignRight)  # 设置整个表格在右面  右对齐
        textTableFormat.setCellPadding(3)  # 内边距
        textTableFormat.setCellSpacing(5)  # 外边距

        # 限制列宽
        # textTableFormat.setColumnWidthConstraints((QTextLength,QTextLength,QTextLength))
        QTextLength
        textTableFormat.setColumnWidthConstraints((QTextLength(QTextLength.PercentageLength, 50), \
                                                   QTextLength(QTextLength.PercentageLength, 40), \
                                                   QTextLength(QTextLength.PercentageLength, 10)))
        # 限制列宽 分别是 50%  40%  10%

        ############################QTextTableFormat 的设置###############################

        textTable = cursor_obj.insertTable(5, 3, textTableFormat)

        QTextTable
        # 插入之后，它会返回 QTextTable 的对象 ，里面存储着关于表格的信息
        # textTable.appendColumns(2)  # 追加了两列  ，详细查看文档

    ############################文本光标对象方法之插入表格###############################


if __name__ == '__main__':
    app = QApplication(sys.argv)

    window = Window()
    window.show()

    sys.exit(app.exec_())