from PyQt5.Qt import *  # 刚开始学习可以这样一下导入
import sys


class Window(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("QMessageBox的学习")
        self.resize(400, 400)
        self.set_ui()

    def set_ui(self):
        messageBox = QMessageBox(QMessageBox.Warning, "xx1", "<h1>xx2</h1>", QMessageBox.Ok | QMessageBox.Discard, self)

        ############################如何比对  标准的按钮###############################
        #获取按钮
        ok_btn= messageBox.button(QMessageBox.Ok)  # 得到真正的按钮对象
        discard_btn= messageBox.button(QMessageBox.Discard)
        def clickedButton_slot(btn):
            if btn == ok_btn:
                print("点击的是ok")
            elif  btn == discard_btn:
                print("点击的是discard")
        messageBox.buttonClicked.connect(clickedButton_slot)
        ############################如何比对  标准的按钮####




        messageBox.show()


if __name__ == '__main__':
    app = QApplication(sys.argv)

    window = Window()
    window.show()

    sys.exit(app.exec_())