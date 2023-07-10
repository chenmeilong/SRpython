
from PyQt5.QtWidgets import QWidget, QCheckBox, QApplication
from PyQt5.QtCore import Qt
import sys

class Example(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        cb = QCheckBox('改改改', self)
        cb.move(20, 20)
        cb.toggle()
        cb.stateChanged.connect(self.changeTitle)   #stateChanged 事件

        self.setGeometry(300, 300, 250, 150)
        self.setWindowTitle('真正的标题')
        self.show()


    def changeTitle(self, state):
        if state == Qt.Checked:
            self.setWindowTitle('假装有标题')
        else:
            self.setWindowTitle('没了')


app = QApplication(sys.argv)
ex = Example()
sys.exit(app.exec_())
