from PyQt5.Qt import * #刚开始学习可以这样一下导入
import sys

class Window(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("QKeySequenceEdit 控件的学习")
        self.resize(400,400)
        self.set_ui()


    def set_ui(self):
        ############################创建QKeySequenceEdit 控件来采集快捷键###############################
        keySequenceEdit = QKeySequenceEdit(self)

        keySequence   =  QKeySequence("Ctrl+C")
        #2 keySequence = QKeySequence(QKeySequence.Copy)
        #3 keySequence = QKeySequence(Qt.CTRL+Qt.Key_C,Qt.CTRL+Qt.Key_C)
        keySequenceEdit.setKeySequence(keySequence)

        keySequenceEdit.editingFinished.connect(lambda :print("结束编辑"))

        keySequenceEdit.keySequenceChanged.connect(lambda arg:print("键位序列发生改变",arg.toString()))

        ############################获取QKeySequenceEdit 中的快捷键###############################
        btn = QPushButton(self )
        btn.setText("按钮")
        btn.move(0,300)

        ############################转化为可读字符串 以及统计  快捷键个数  ###############################
        btn.clicked.connect(lambda :print(keySequenceEdit.keySequence().toString()))
        btn.clicked.connect(lambda :print(keySequenceEdit.keySequence().count()))

        #清除
        # keySequenceEdit.clear()
        ############################转化为可读字符串 以及统计  快捷键个数 ###############################

        ############################获取QKeySequenceEdit 中的快捷键###############################

        ############################创建QKeySequenceEdit 控件来采集快捷键###############################

if __name__ == '__main__':
    app =QApplication(sys.argv)

    window = Window()
    window.show()

    sys.exit(app.exec_())