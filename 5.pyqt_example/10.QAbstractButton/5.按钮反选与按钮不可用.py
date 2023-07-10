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
push_button.setCheckable(True)

radio_button = QRadioButton(window)
radio_button.setText("这是QRadioButton")
radio_button.move(100,100)

check_box = QCheckBox(window)
check_box.setText("这是QCheckBox")
check_box.move(100,150)

push_button.setDown(True)
radio_button.setDown(True)
check_box.setDown(True)

############################切换按钮###############################
btn = QPushButton(window)
btn.setText("切换状态按钮")

def btn_pressed_slot():
    #法一
    push_button.toggle()
    radio_button.toggle()
    check_box.toggle()
    #法二
    # push_button.setChecked(not push_button.isChecked())
    # radio_button.setChecked(not radio_button.isChecked())
    # check_box.setChecked(not check_box.isChecked())

btn.pressed.connect(btn_pressed_slot)

check_box.setEnabled(False )    #设置为不可用
############################切换按钮###############################


#展示控件
window.show()

#3,进入消息循环
sys.exit(app.exec_())