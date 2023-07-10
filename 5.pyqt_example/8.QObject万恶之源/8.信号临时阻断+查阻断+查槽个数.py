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
        self.obj.setObjectName("对象0")
        def objectNameChanged_slot(arg):  # 会自动接收信号发送的参数
            print("对象的名字被修改成了{}".format(arg))
        self.obj.objectNameChanged.connect(objectNameChanged_slot)

        self.obj.setObjectName("tom")  #触发

        self.obj.blockSignals(True)  # 临时阻断self.obj所有信号的连接（注不是断开连接）
        print(self.obj.signalsBlocked())

        self.obj.setObjectName("jack")  #不触发
        self.obj.blockSignals(False)  # 将临时阻断去掉
        print(self.obj.signalsBlocked())  #False 表示没有被临时阻断  ######################################
        self.obj.setObjectName("richel")   #也想触发

        print("信号有{}个槽函数".format(self.obj.receivers(self.obj.objectNameChanged)))     ## 利用receivers() 查看指定信号所对应的槽函数的个数

if __name__ == '__main__':
    app = QApplication(sys.argv)

    window = Window()
    window.show()

    sys.exit(app.exec_())
# 使用blockSignals()进行临时阻断的设置