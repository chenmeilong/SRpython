#一个对象只能设置一个父对象，而且是按后设置的算！

from PyQt5.Qt import *  # 刚开始学习可以这样一下导入
import sys


class Window(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("QObject的学习")
        self.resize(400, 400)
        self.set_ui()
    def set_ui(self):
        self.QObject_test()
    def QObject_test(self):
        obj0 = QObject()
        obj1 = QObject()
        obj2 = QObject()
        obj3 = QObject()
        obj4 = QObject()
        obj5 = QObject()

        str_pr ="obj"
        for i in range(6):  # 打印各个对象变量
            name = str_pr+str(i)
            print(eval(name))                                         #字符串转成变量

        obj1.setParent(obj0)
        obj2.setParent(obj0)
        obj2.setObjectName("2")
        obj3.setParent(obj1)

        obj4.setParent(obj2)
        obj5.setParent(obj2)

        print(obj0.children())  #注：这个子对象是直接的子对象。


        print(obj0.findChild(QObject,"2"))   #第二个参数为对象的名称    找子对象

if __name__ == '__main__':
    app = QApplication(sys.argv)

    window = Window()
    window.show()

    sys.exit(app.exec_())

