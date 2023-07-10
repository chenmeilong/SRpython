from PyQt5.Qt import * #刚开始学习可以这样一下导入
import sys
#1,创建app
app  = QApplication(sys.argv)


#2，控件的操作：
#创建控件
window = QWidget()


#设置控件
window.setWindowTitle("QCheckBox")
window.resize(500,500)

checkbox1 = QCheckBox(window)
checkbox1.setText("Python")
checkbox1.setIcon(QIcon("icon.ico"))
checkbox1.setShortcut("Ctrl+P")
checkbox1.setTristate(True)  #设置三种状态

###########################################################
checkbox1.stateChanged.connect(lambda state:print(state))
###########################################################

checkbox2 = QCheckBox(window)
checkbox2.setText("C++")
checkbox2.move(0,30)

checkbox3 = QCheckBox(window)
checkbox3.setText("C")
checkbox3.move(0,60)



#展示控件
window.show()

#3,进入消息循环
sys.exit(app.exec_())