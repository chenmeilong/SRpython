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

############################是否被编辑###############################
def btn_pressed_slot():
    print(lineEdit.isModified())  #输出是否被编辑
    lineEdit.setModified(False)   #清空编辑记录

############################是否被编辑###############################

btn = QPushButton(window)
btn.setText("按钮")
btn.move(0,300)
btn.pressed.connect(btn_pressed_slot)


#展示控件
window.show()

#3,进入消息循环
sys.exit(app.exec_())