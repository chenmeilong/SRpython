from PyQt5.Qt import * #刚开始学习可以这样一下导入
import sys

class Window(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("QErrorMessage的学习")
        self.resize(400,400)
        self.set_ui()


    def set_ui(self):
        errorMessage = QErrorMessage(self)

        #窗口标题
        errorMessage.setWindowTitle("错误提示")

        # QErrorMessage.qtHandler()
        # qDebug("xxx")
        # qWarning("sdfsdf")

        errorMessage.showMessage("Life is short ,I learn Python")  #注，showMessage() 会自动弹出
        errorMessage.showMessage("第二个框")



        # errorMessage.exec()
        errorMessage.open()
        # errorMessage.show()


if __name__ == '__main__':
    app =QApplication(sys.argv)

    window = Window()
    window.show()

    sys.exit(app.exec_())