
import sys
from PyQt5.QtWidgets import QMainWindow
from PyQt5.QtWidgets import QApplication
from PyQt5.QtWidgets import QAction
from PyQt5.QtWidgets import qApp
from PyQt5.QtGui import QIcon

class Ex(QMainWindow):

    def __init__(self):
        super(Ex, self).__init__()
        self.initUI()

    def initUI(self):
        exitAct = QAction(QIcon("exit.png"),'&Exit',self)
        print(exitAct)
        exitAct.setShortcut("ctrl+q")          #添加快捷键
        exitAct.triggered.connect(qApp.quit)     #添加绑定事件

        exitAct.setStatusTip('tuichu应用')  # 添加状态栏提示   使用前需要创建状态栏
        self.statusBar()         #创建一个状态栏  返回一个状态栏对象

        menubar = self.menuBar()    #创建菜单栏
        fileMenu = menubar.addMenu('&File')
        fileMenu.addAction(exitAct)

        self.setGeometry(300,300,399,200)
        self.setWindowTitle('这是标题')
        self.show()

app = QApplication(sys.argv)
demo1 = Ex()
sys.exit(app.exec_())

