import sys
from PyQt5.QtWidgets import QMainWindow, QPushButton, QApplication
class Example(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        btn1 = QPushButton("按钮老大", self)
        btn1.move(30, 50)

        btn2 = QPushButton("按钮老二", self)
        btn2.move(150, 50)

        btn1.clicked.connect(self.buttonClicked)
        btn2.clicked.connect(self.buttonClicked)

        self.statusBar()

        self.setGeometry(300, 300, 290, 150)
        self.setWindowTitle('事件发送')
        self.show()

    def buttonClicked(self):
        sender = self.sender()   #指触发事件的控件，之所以是object类型，是因为button按钮也是一个类
        self.statusBar().showMessage(sender.text() + '被按那儿了')

app = QApplication(sys.argv)
ex = Example()
sys.exit(app.exec_())
