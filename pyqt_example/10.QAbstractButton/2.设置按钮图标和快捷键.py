from PyQt5.Qt import * #刚开始学习可以这样一下导入
import sys
#1,创建app
app = QApplication(sys.argv)

#2，控件的操作：
#创建控件
window = QWidget()

#设置控件
window.setWindowTitle("QAbstractButton提示文本")
window.resize(500,500)

btn = QPushButton(window)
btn.setText("1")
btn.setShortcut("Ctrl+a")
def btn_pressed_slot():
    num = int(btn.text())+1
    btn.setText(str(num))

btn.pressed.connect(lambda :print("btn2被点"))  # 用clicked 也可以

# btn.pressed.connect(btn_pressed_slot)  # 用clicked 也可以  #这种方法有点问题，因为快捷键只能触发一次
############################图标相关API###############################
icon = QIcon("icon.ico")
btn.setIcon(icon)

size = QSize(50,50)
btn.setIconSize(size)

#获取图标
print(btn.icon())  #<PyQt5.QtGui.QIcon object at 0x000001D7C7CB3B88>
print(btn.iconSize())  #PyQt5.QtCore.QSize(50, 50)

############################图标相关API###############################
#展示控件
window.show()

#3,进入消息循环
sys.exit(app.exec_())