
import sys
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QWidget, QApplication
class Example(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
    def initUI(self):
        self.setGeometry(300, 300, 250, 150)
        self.setWindowTitle('事件的手')
        self.show()

    def keyPressEvent(self, e):
        if e.key() == Qt.Key_Escape:
            self.close()

app = QApplication(sys.argv)
ex = Example()
sys.exit(app.exec_())
