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
radio_button_1.setChecked(True)

radio_button_2 = QRadioButton("女-Famale",window)
radio_button_2.move(100,200)
radio_button_2.setIcon(QIcon("icon.ico"))
radio_button_2.setShortcut("Ctrl+F")

###########################################################
sex_group = QButtonGroup(window)
sex_group.addButton(radio_button_1,1)  #绑定id=1
sex_group.addButton(radio_button_2,2)  #id=2
###########################################################


radio_button_yes = QRadioButton("yes",window)
radio_button_yes.move(300,100)

radio_button_no = QRadioButton("yes",window)
radio_button_no.move(300,200)
###########################################################
judge_group = QButtonGroup(window)
judge_group.addButton(radio_button_yes)
judge_group.addButton(radio_button_no)

###########################################################


#****************************查看组中的按钮*******************************
print(sex_group.buttons())
print(judge_group.buttons())
#****************************查看组中的按钮*******************************

#****************************查看id为 1 的按钮*******************************
print(sex_group.button(1))
#****************************查看id为 1 的按钮*******************************

#****************************查看组中被选中的按钮*******************************
print(sex_group.checkedButton())
#****************************查看组中被选中的按钮******************************


#移除按钮    它并不是从window 上删除这个按钮，而只是将其移出抽象的按钮组。
#****************************移出按钮组*******************************
sex_group.removeButton(radio_button_2)
#****************************移出按钮组*******************************

#****************************获取id *******************************
print(sex_group.id(radio_button_1))
#****************************查看当前选中的按钮的id****
print(sex_group.checkedId())


#单选变多选
#****************************将一个组的独占设置为否定*******************************
# sex_group.setExclusive(False)
#****************************将一个组的独占设置为否定*******************************


#展示控件
window.show()
#3,进入消息循环
sys.exit(app.exec_())