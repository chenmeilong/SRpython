from PyQt5.Qt import * #刚开始学习可以这样一下导入
import sys

#1,创建app
app  = QApplication(sys.argv)

#2，控件的操作：
#创建控件
window = QWidget()

#设置控件
window.setWindowTitle("QAbstactButton 有效区域")
window.resize(500,500)

btn = QPushButton(window)
btn.setText("xxx")
btn.setIcon(QIcon("icon.ico"))

btn2 = QPushButton(window)
btn2.setText("btn2")
btn2.move(200,200)

############################默认处理###############################
btn2.setAutoDefault(True)  #鼠标点击之后再会被设置为默认

print(btn.autoDefault())  #false
print(btn2.autoDefault())  #true
############################默认处理###############################

#展示控件
window.show()

#3,进入消息循环
sys.exit(app.exec_())
