from PyQt5.Qt import * #刚开始学习可以这样一下导入
import sys
#1,创建app
app  = QApplication(sys.argv)

#2，控件的操作：
#创建控件
window = QWidget()

#设置控件
window.setWindowTitle("QAbstractButton提示文本")
window.resize(500,500)

btn = QPushButton(window)
num = 0
def btn_pressed_slot():
    global num
    num =num + 1
    print(num)

btn.pressed.connect(btn_pressed_slot)

icon = QIcon("icon.ico")
btn.setIcon(icon)

size = QSize(50,50)
btn.setIconSize(size)

############################自动重复相关###############################
#查看是否自动重复
print(btn.autoRepeat())
#设置自动重复
btn.setAutoRepeat(True)
btn.setAutoRepeatDelay(500)  #初次检测延迟为2s
btn.setAutoRepeatInterval(500)  #重复检测间隔为1s

print(btn.autoRepeatDelay())   #首次延迟
print(btn.autoRepeatInterval()) #以后的触发间隔

#展示控件
window.show()
#3,进入消息循环
sys.exit(app.exec_())