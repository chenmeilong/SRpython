
import sys
from PyQt5.QtWidgets import QMainWindow, QAction, QApplication

class Example(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.statusbar = self.statusBar()
        self.statusbar.showMessage('Ready')

        menubar = self.menuBar()
        viewMenu = menubar.addMenu('View')

        # 本例创建了一个行为菜单。这个行为／动作能切换状态栏显示或者隐藏。
        viewStatAct = QAction('View status', self, checkable=True)
        viewStatAct.setStatusTip('View statusbar')        # 用checkable选项创建一个能选中的菜单。
        viewStatAct.setChecked(True)        # 默认设置为选中状态
        viewStatAct.triggered.connect(self.toggleMenu)
        print(viewStatAct)

        viewMenu.addAction(viewStatAct)
        # 依据选中状态切换状态栏的显示与否。
        self.setGeometry(300, 300, 300, 200)
        self.setWindowTitle('Check menu')


    def toggleMenu(self, state):
        if state:
            self.statusbar.show()
        else:
            self.statusbar.hide()

app = QApplication(sys.argv)
demo1 = Example()
demo1.show()
sys.exit(app.exec_())
