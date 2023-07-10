from PyQt5.Qt import * #刚开始学习可以这样一下导入
import sys
#1,创建app
app  = QApplication(sys.argv)
#2，控件的操作：
#创建控件
window = QWidget()
#设置控件
window.setWindowTitle("QLineEdit的学习")
window.resize(500,500)

############################案例###############################
lineEdit_1 = QLineEdit(window)
lineEdit_1.move(100,100)

lineEdit_2 = QLineEdit(window)
lineEdit_2.move(100,200)

copy_btn = QPushButton(window)
copy_btn.setText("复制")
copy_btn.move(100,300)

# copy_btn.pressed  和 copy_btn.clicked
#二者不完全一样， pressed 是按下即可，  clicked 是按下加松开
def copy_btn_slot():
    content = lineEdit_1.text()
    lineEdit_2.setText(content)

copy_btn.pressed.connect(copy_btn_slot)
############################案例###############################

#展示控件
window.show()
#3,进入消息循环
sys.exit(app.exec_())