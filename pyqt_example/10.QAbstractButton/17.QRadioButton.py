from PyQt5.Qt import * #刚开始学习可以这样一下导入
import sys
#1,创建app
app  = QApplication(sys.argv)


#2，控件的操作：
#创建控件
window = QWidget()


#设置控件
window.setWindowTitle("QRadioButton")
window.resize(500,500)


radio_button_1 = QRadioButton("男-Male",window)
radio_button_1.move(100,100)
radio_button_1.setIcon(QIcon("icon.ico"))
radio_button_1.setShortcut("Ctrl+M")
# radio_button_1.setChecked(True)  # 它是默认选中


radio_button_2 = QRadioButton("女-Famale",window)
radio_button_2.move(100,200)
radio_button_2.setIcon(QIcon("icon.ico"))
radio_button_2.setShortcut("Ctrl+F")

radio_button_2.toggled.connect(lambda :print("女 发送状态切换"))

# radio_button_1.setAutoExclusive(False)  # 此时将第一个 拿出来了，另外两个可单选


#展示控件
window.show()

#3,进入消息循环
sys.exit(app.exec_())