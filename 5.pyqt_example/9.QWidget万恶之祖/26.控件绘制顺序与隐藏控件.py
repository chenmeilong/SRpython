from PyQt5.Qt import * #刚开始学习可以这样一下导入
import sys

class Window(QWidget):
    def paintEvent(self, event):  #窗口绘制事件
        print("窗口被绘制了")
        return super().paintEvent(event)

class Btn(QPushButton):
    def paintEvent(self, event):
        print("里面控件被绘制了")
        return super().paintEvent(event)


#1,创建app
app  = QApplication(sys.argv)

#2，控件的操作：
#创建控件
window = Window()

#设置控件
window.setWindowTitle("交互状态")
window.resize(500,500)

btn = Btn(window)
btn.setText("按钮")
btn.pressed.connect(lambda :btn.setVisible(False))  #点击之后，就会把它给隐藏了，
                                                    # 后面的绘制就不会显示它了，但是这个对象还是存在的
#隐藏这个按钮的四种方法
# btn.setVisible(False)
# btn.setHidden(True)
# btn.hide()
# btn.close()

# 使用这种办法释放控件
#btn.setAttribute(Qt.WA_DeleteOnClose,True)  # 这再调用close()就会释放按钮了
# btn.close()

#展示控件
# window.show()
# window.setVisible(True)
window.setHidden(False)  #它也可以绘制window

#3,进入消息循环
sys.exit(app.exec_())
