import sys
from PyQt5.QtWidgets import QMainWindow, QAction, QMenu, QApplication
class Example(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()
    def initUI(self):
        menubar = self.menuBar()
        fileMenu = menubar.addMenu('File')

        impMenu = QMenu('Import', self)        #创建菜单容器   可以用在右键菜单栏
        impAct = QAction('Import mail', self)
        impMenu.addAction(impAct)     #添加子菜单

        newAct = QAction('New', self)
        fileMenu.addAction(newAct)
        fileMenu.addMenu(impMenu)
        self.setGeometry(300, 300, 300, 200)
        self.setWindowTitle('Submenu')
        self.show()
app = QApplication(sys.argv)
ex = Example()
sys.exit(app.exec_())
