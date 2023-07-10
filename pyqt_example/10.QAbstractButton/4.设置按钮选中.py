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

push_button = QPushButton(window)
push_button.setText("这是QPushButton")
push_button.move(100,50)
push_button.setStyleSheet("QPushButton:pressed {background-color:red;}")

radio_button = QRadioButton(window)
radio_button.setText("这是QRadioButton")
radio_button.move(100,100)

check_box = QCheckBox(window)
check_box.setText("这是QCheckBox")
check_box.move(100,150)

# push_button.setDown(True)    #设置按钮是否 被按下
radio_button.setDown(True)
check_box.setDown(True)

############################是否被选中以及设置选中它###############################

print(push_button.isCheckable())  #False      能否被选中
print(radio_button.isCheckable()) #True
print(check_box.isCheckable())    #True

print(push_button.isChecked())    #是否被选中
print(radio_button.isChecked())
print(check_box.isChecked())

push_button.setCheckable(True)  # 先要让它能够被选中  这样按钮就会有开关效果

push_button.setChecked(True)
radio_button.setChecked(True)
check_box.setChecked(True)

############################是否被选中以及设置选中它###############################

#展示控件
window.show()

#3,进入消息循环
sys.exit(app.exec_())