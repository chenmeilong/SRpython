from PyQt5.Qt import * #刚开始学习可以这样一下导入
import sys

class Window(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("QMessageBox的学习")
        self.resize(400,400)
        self.set_ui()


    def set_ui(self):
        messageBox = QMessageBox(self)

        messageBox.setText("<h1>主要文本</h1>")
        messageBox.setDetailedText("Life is short ,I leran Python")
        messageBox.setInformativeText("提示文本")
        messageBox.setWindowTitle("消息提示")


        messageBox.addButton(QPushButton("xx1",messageBox),QMessageBox.YesRole)
        messageBox.addButton(QPushButton("xx2",messageBox),QMessageBox.NoRole)


        def clickedButton_slot(btn):
            #先拿到role
            role = messageBox.buttonRole(btn)
            if role == QMessageBox.YesRole:
                print("点击的是第一个")
            elif role == QMessageBox.NoRole:
                print("点击的是第二个")

        messageBox.buttonClicked.connect(clickedButton_slot)

        #文本交互  它主要控制的是主标题
        messageBox.setTextInteractionFlags(Qt.TextEditorInteraction)


        messageBox.show()

if __name__ == '__main__':
    app =QApplication(sys.argv)

    window = Window()
    window.show()

    sys.exit(app.exec_())