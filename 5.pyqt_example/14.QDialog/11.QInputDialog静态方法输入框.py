from PyQt5.Qt import * #刚开始学习可以这样一下导入
import sys

class Window(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("QInputDialog的学习")
        self.resize(400,400)
        self.set_ui()


    def set_ui(self):
        # ret = QInputDialog.getInt(self,"请输入一个值","Hello World",0,step= 8)
        # ret = QInputDialog.getDouble(self,"请输入一个值","Hello World",0.0,decimals = 3)
        # ret = QInputDialog.getText(self,"请输入","Hello World",echo=QLineEdit.Password)
        # ret = QInputDialog.getMultiLineText(self,"请输入","Hello World","default")
        ret = QInputDialog.getItem(self,"请输入","Hello World",["1","2","3"],2,True)

        print(ret)


if __name__ == '__main__':
    app =QApplication(sys.argv)

    window = Window()
    window.show()

    sys.exit(app.exec_())