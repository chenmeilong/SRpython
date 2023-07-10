from PyQt5.Qt import * #刚开始学习可以这样一下导入
import sys
#1,创建app
app  = QApplication(sys.argv)


#2，控件的操作：
#创建控件
window = QWidget()


#设置控件
window.setWindowTitle("QLineEdit")
window.resize(500,500)

############################middle###############################
widget_w = 150
widget_h = 40
widget_h_margin = 10
top_margin = 30

account_lineEdit = QLineEdit(window)
account_lineEdit.resize(widget_w,widget_h)

pwd_lineEdit = QLineEdit(window)
pwd_lineEdit.resize(widget_w,widget_h)

btn = QPushButton(window)
btn.setText("按钮")
btn.resize(widget_w,widget_h)

x = (window.width() - widget_w)/2
y1 = top_margin
y2 = y1 + widget_h +widget_h_margin
y3 = y2 + widget_h +widget_h_margin

account_lineEdit.move(x,y1)
pwd_lineEdit.move(x,y2)
btn.move(x,y3)

############################middle###############################

############################输入限制###############################
#1  最大长度设置
account_lineEdit.setMaxLength(3)  # 字符个数的限制
print(account_lineEdit.maxLength())  #获取最大长度
#2  设置只读
account_lineEdit.setReadOnly(True)  #设置只读

account_lineEdit.setText("hello ")  #只读是可以用代码来设置的！
print(account_lineEdit.isReadOnly()) #查看是否只读

############################输入限制###############################


#展示控件
window.show()

#3,进入消息循环
sys.exit(app.exec_())