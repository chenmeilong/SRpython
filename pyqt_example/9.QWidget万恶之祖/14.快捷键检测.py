from PyQt5.Qt import * #刚开始学习可以这样一下导入
import sys


class MyLabel(QLabel):
    def enterEvent(self, *args, **kwargs):
        self.setText("欢迎光临")
        print("鼠标进入")
    def leaveEvent(self, *args, **kwargs):
        self.setText("谢谢惠顾")
        print("鼠标离开")

    # def keyPressEvent(self, QKeyEvent):
    #
    def keyPressEvent(self, event):
        # event.key() == Qt.Key_Tab   #所有的普通都可以这样对比
        #键分为  普通键和修饰键  ctrl alt fn 等键
        if event.key() == Qt.Key_Tab:
            print("点击了Tab键")

        #modifiers()后面是按位与运算
        if event.modifiers() == Qt.ControlModifier | Qt.ShiftModifier  and event.key() == Qt.Key_A:  #modifiers() 是获取修饰键的
            print("用户点击了 Ctrl + Shift+ A ")


#1,创建app
app  = QApplication(sys.argv)


#2，控件的操作：
#创建控件
window = QWidget()


#设置控件
window.setWindowTitle("事件的案例1")
window.resize(500,500)

label = MyLabel(window)
label.setStyleSheet("background-color:cyan")
label.move(200,200)
label.resize(100,100)
label.grabKeyboard()          # 让label 捕获键盘


#展示控件
window.show()

#3,进入消息循环
sys.exit(app.exec_())