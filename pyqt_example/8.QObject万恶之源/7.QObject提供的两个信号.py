# obj.destroyed  #当对象被销毁时会触发这个信号
# obj.objectNameChanged # 当对象的名称改变时触发这个信号

from PyQt5.Qt import *  # 刚开始学习可以这样一下导入
import sys

class Window(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("QObject的学习")
        self.resize(400, 400)
        self.set_ui()
    def set_ui(self):
        self.QObject_signal_test()
    def QObject_signal_test(self):
        self.obj = QObject()
        def destrroy_slot(arg):
            print(arg)
            print("对象被释放了")
        def nameChanged_slot(arg):
            print(arg)
            # print(arg.objectName())  #可以得到对象的名字 ，前提需要设置对象的名字
            print("对象名称被修改了")
        self.obj.destroyed.connect(destrroy_slot)

        self.obj.objectNameChanged.connect(nameChanged_slot)
        self.obj.setObjectName("zcb")

        self.obj.objectNameChanged.disconnect()  # 取消信号和槽之间的联系
        print("取消信号和槽之间的联系")
        self.obj.setObjectName("tom")

        del self.obj

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = Window()
    window.show()
    sys.exit(app.exec_())

