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


lineEdit = QLineEdit("hello",window)  #一行搞定下面两行
# lineEdit = QLineEdit(window)
# lineEdit.setText("hello ")

#插入 文本  (它是在光标之后插入)

btn = QPushButton(window)
btn.setText("按钮")
btn.move(100,100)

btn.pressed.connect(lambda :lineEdit.insert("I learn Python"))
#2
#获取文本  获取的是真实的内容
print(lineEdit.text())


#3
#获取用户看到的内容
print(lineEdit.displayText())  #例如加密的时候这里看到的就是*****

############################文本的设置和获取###############################

#展示控件
window.show()
#3,进入消息循环
sys.exit(app.exec_())