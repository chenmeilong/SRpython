import sys
# from PyQt5.QtWidgets import QWidget
# from PyQt5.QtWidgets import QDesktopWidget
# QDesktopWidget这个库提供了用户的桌面信息,包括屏幕的大小
from PyQt5.QtWidgets import QApplication
from PyQt5.QtWidgets import QMainWindow

class Ex(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        # 状态栏是由这个创建的
        self.statusBar().showMessage('准备')   #3 showMessage()方法在状态栏上显示一条信息
        # 调用QtGui.QMainWindow 类的 statusBar()方法
        #3 创建状态栏.第一次调用创建一个状态栏,返回一个状态栏对象.
        self.setGeometry(300,300,250,150)            #屏幕位置和长和宽
        self.setWindowTitle('标题还是要取的')
        #显示
        self.setStatusTip('tuichu应用')  #鼠标移植当前self对象时显示该  状态信息
        self.show()

app = QApplication(sys.argv)
demo1 = Ex()
sys.exit(app.exec_())

