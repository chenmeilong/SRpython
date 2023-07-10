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

############################QPushButton的构造函数###############################
#两种方式：
# 第一种
    # btn = QPushButton()
    # btn.setParent(window)
    # btn.setText("xxx")
    # btn.setIcon(QIcon("icon.ico"))
#第二种：
# btn = QPushButton(QIcon("icon.ico"),"xxx",window)  #一步搞定

#我喜欢的方式：
btn = QPushButton(window)
btn.setText("xxx")
btn.setIcon(QIcon("icon.ico"))
############################QPushButton的构造函数###############################

#展示控件
window.show()
#3,进入消息循环
sys.exit(app.exec_())