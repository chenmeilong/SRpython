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


radio_button_2 = QRadioButton("女-Famale",window)
radio_button_2.move(100,200)
radio_button_2.setIcon(QIcon("icon.ico"))
radio_button_2.setShortcut("Ctrl+F")
radio_button_1.setChecked(True)
sex_group = QButtonGroup(window)
sex_group.addButton(radio_button_1,1)
sex_group.addButton(radio_button_2,2)



# sex_group.buttonClicked.connect(lambda val:print(val))  #信号 向外传出的是具体的按钮
# sex_group.buttonClicked[int].connect(lambda val:print(val))  #添加按钮的时候给它设置id

sex_group.buttonClicked.connect(lambda val:print(val,sex_group.id(val)))  #按钮和id


radio_button_yes = QRadioButton("yes",window)
radio_button_yes.move(300,100)

radio_button_no = QRadioButton("yes",window)
radio_button_no.move(300,200)

###########################################################
judge_group = QButtonGroup(window)
judge_group.addButton(radio_button_yes)
judge_group.addButton(radio_button_no)
###########################################################



#展示控件
window.show()

#3,进入消息循环
sys.exit(app.exec_())