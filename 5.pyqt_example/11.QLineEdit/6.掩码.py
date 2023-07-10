from PyQt5.Qt import * #刚开始学习可以这样一下导入
import sys
#1,创建app
app  = QApplication(sys.argv)


#2，控件的操作：
#创建控件
window = QWidget()

#设置控件
window.setWindowTitle("掩码")
window.resize(500,500)

lineEdit = QLineEdit(window)
lineEdit.move(100,100)

############################设置掩码###############################
#1总共有5位， 左边2 （必须是大写字母） 中间是 -  分隔符 右边两位（必须是数字）

# lineEdit.setInputMask(">AA-99")
#2 设置占位字符
lineEdit.setInputMask(">AA-99;#")   #掩码字符

#如果需要占位字符就要用;  后面加上字符即可

############################设置掩码###############################

#展示控件
window.show()

#3,进入消息循环
sys.exit(app.exec_())