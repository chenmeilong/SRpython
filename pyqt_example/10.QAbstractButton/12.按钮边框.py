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

############################边框是否扁平化###############################
btn.setStyleSheet("background-color:red;") #扁平化使背景颜色不再绘制
print(btn.isFlat())  #false
btn.setFlat(True)  # 这时的按钮就不再凸起了
############################边框是否扁平化###############################

#展示控件
window.show()
#3,进入消息循环
sys.exit(app.exec_())