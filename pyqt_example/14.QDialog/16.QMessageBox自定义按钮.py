from PyQt5.Qt import *  # 刚开始学习可以这样一下导入
import sys


class Window(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("QMessageBox的学习")
        self.resize(400, 400)
        self.set_ui()

    def set_ui(self):
        messageBox = QMessageBox(self)
        # messageBox = QMessageBox(QMessageBox.Warning, "xx1", "<h1>xx2</h1>", QMessageBox.Ok | QMessageBox.Discard, self)

        # # 强行变为非模态的方法：
        # messageBox.setModal(False)
        # messageBox.setWindowModality(Qt.NonModal)


        #识别自定义按钮
        btn_1= QPushButton("xx1",messageBox)
        btn_2 = QPushButton("xx2",messageBox)
        messageBox.addButton(btn_1,QMessageBox.YesRole)
        messageBox.addButton(btn_2,QMessageBox.NoRole)
        def clickedButton_slot(btn):
            if btn == btn_1:
                print("点击的是第一个按钮")

            elif btn== btn_2:
                print("点击的是第二个按钮")
        messageBox.buttonClicked.connect(clickedButton_slot)




        messageBox.show()


if __name__ == '__main__':
    app = QApplication(sys.argv)

    window = Window()
    window.show()

    sys.exit(app.exec_())