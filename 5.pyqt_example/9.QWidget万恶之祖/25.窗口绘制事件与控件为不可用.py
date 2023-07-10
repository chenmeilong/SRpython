from PyQt5.Qt import * #刚开始学习可以这样一下导入
import sys

class Window(QWidget):
    def paintEvent(self, event):  #窗口绘制事件
        print("窗口被绘制了")
        return super().paintEvent(event)

#1,创建app
app  = QApplication(sys.argv)
#2，控件的操作：
#创建控件
window = Window()

#设置控件
window.setWindowTitle("交互状态")
window.resize(500,500)


btn = QPushButton(window)
btn.setText("按钮")
btn.pressed.connect(lambda :print("点击按钮"))
btn.setEnabled(False)  #将它设置按钮控件为不可用
print(btn.isEnabled())  # 查看它是否可用


#展示控件  3种方法中的任意一个都行
window.show()
# window.setVisible(True)
# window.setHidden(False)  #它也可以绘制window


#3,进入消息循环
sys.exit(app.exec_())