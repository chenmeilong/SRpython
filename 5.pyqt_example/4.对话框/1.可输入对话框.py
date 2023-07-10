
from PyQt5.QtWidgets import (QWidget, QPushButton, QLineEdit,
    QInputDialog, QApplication)
import sys

class Example(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.btn = QPushButton('Dialog', self)
        self.btn.move(20, 20)
        self.btn.clicked.connect(self.showDialog)

        self.le = QLineEdit(self)
        self.le.move(130, 22)

        self.setGeometry(300, 300, 290, 150)
        self.setWindowTitle('Input dialog')
        self.show()


    def showDialog(self):
        text, ok = QInputDialog.getText(self, 'Input Dialog','Enter your name:')
        if ok:
            self.le.setText(str(text))
        print(text+"哈哈")

app = QApplication(sys.argv)
ex = Example()
sys.exit(app.exec_())
