from PyQt5.Qt import *  # 刚开始学习可以这样一下导入
import sys


class App(QApplication):
    # def notify(self, QObject, QEvent):  #参数：一个是事件的接收者，一个是 被包装的事件对象
    #     pass
    QEvent
    def notify(self, receiver, event):

        if receiver.inherits("QPushButton") and event.type() ==  QEvent.MouseButtonPress:
            print(receiver, event)  #此时 pressed 信号就不会被发出去了

        return super().notify(receiver, event)


class Btn(QPushButton):

    # def event(self, QEvent):

    def event(self, event):  # 继续往下分发
        if event.type() == QEvent.MouseButtonPress:
            print(event)
        return super().event(event)   # 继续往下分发
    def mousePressEvent(self, *args, **kwargs):
        print("请注意我，我不是I am here")
        super().mousePressEvent(*args,**kwargs)


app = App(sys.argv)
window = QWidget()

btn = Btn(window)
btn.setText("按钮")
btn.move(100, 100)

btn.pressed.connect(lambda: print("I am here"))
# 鼠标只要是按下就行，而btn.clicked  鼠标是点击之后又起来


window.show()

sys.exit(app.exec_())

'''
    输出：
    <__main__.Btn object at 0x0000020A212B1D38> <PyQt5.QtGui.QMouseEvent object at 0x0000020A212B1EE8>
    <PyQt5.QtGui.QMouseEvent object at 0x0000020A212B1EE8>
    请注意我，我不是I am here
    I am here
'''