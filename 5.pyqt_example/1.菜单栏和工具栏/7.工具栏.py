
# 菜单栏包含了所有的命令，工具栏就是常用的命令的集合。

import sys
from PyQt5.QtWidgets import QMainWindow, QAction, qApp, QApplication
from PyQt5.QtGui import QIcon

class Example(QMainWindow):

    def __init__(self):
        super().__init__()
        self.initUI()
    def initUI(self):
        exitAct = QAction(QIcon('exit.png'), 'Exit', self)
        exitAct.setShortcut('Ctrl+Q')
        exitAct.triggered.connect(qApp.quit)
        # 和上面的菜单栏差不多，这里使用了一个行为对象，
        # 这个对象绑定了一个标签，一个图标和一个快捷键。
        # 这些行为被触发的时候，会调用QtGui.QMainWindow的quit方法退出应用。
        self.toolbar = self.addToolBar('Exit')  #设置小提示   添加工具栏
        self.toolbar.addAction(exitAct)

        self.setGeometry(300, 300, 300, 200)
        self.setWindowTitle('Toolbar')
        self.show()


app = QApplication(sys.argv)
ex = Example()
sys.exit(app.exec_())
# 上面的例子中，我们创建了一个工具栏这个工具栏只有一个退出应用的动作

