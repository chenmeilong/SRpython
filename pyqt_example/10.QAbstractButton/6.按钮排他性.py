from PyQt5.Qt import * #刚开始学习可以这样一下导入
import sys
#1,创建app
app  = QApplication(sys.argv)


#2，控件的操作：
#创建控件
window = QWidget()
#设置控件
window.setWindowTitle("QAbstractButton之按钮的状态")
window.resize(500,500)

############################排他性###############################
for i in range(3):
    btn = QPushButton(window)
    btn.setText("btn"+str(i))
    btn.move(50*i,50*i)

    btn.setCheckable(True)  # 先要让它能够被选中
    print(btn.autoExclusive())  #默认都是不排他的

    btn.setAutoExclusive(True)     #设置为排他按钮

#QRadioButton 默认排他
#QCheckBox默认不排他
############################排他性###############################

#展示控件
window.show()
#3,进入消息循环
sys.exit(app.exec_())