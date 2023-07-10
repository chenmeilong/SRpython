
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
        obj1= QObject()
        self.obj = obj1
        obj2 = QObject(obj1)

        #监听obj2
        obj1.destroyed.connect(lambda :print("obj1被释放"))
        self.obj.destroyed.connect(lambda :print("self.obj被释放"))
        obj2.destroyed.connect(lambda :print("obj2被释放"))             #一旦父对象被释放，子对象也自动被释放
        #删除对象
        # del self.obj         #释放obj1  子对象obj2也会被释放

        #删除对象
        # del obj2 # 这时候没有任何效果，因为obj_2 还被obj_1 引用着呢！
        obj2.deleteLater()  # 删除对象时，也会解除它与父对象的关系，而且是稍后删除。

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = Window()
    window.show()
    sys.exit(app.exec_())

