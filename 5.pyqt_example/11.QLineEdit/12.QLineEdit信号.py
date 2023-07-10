from PyQt5.Qt import * #刚开始学习可以这样一下导入
import sys
#1,创建app
app  = QApplication(sys.argv)

#2，控件的操作：
#创建控件
window = QWidget()

#设置控件
window.setWindowTitle("光标")
window.resize(500,500)

lineEdit = QLineEdit(window)
lineEdit.move(100,100)

############################信号###############################
#1
#textEdited 信号
# lineEdit.textEdited.connect(lambda val:print("编辑",val))

#2
#textChanged 信号
# lineEdit.textEdited.connect(lambda val:print("内容变化",val))
# 1 和 2 的区别是，1 是针对用户用键盘编辑的时候触发，而2是只要是内容变化就触发
#3
#returnPressed 信号
#按回车键触发
# lineEdit.returnPressed.connect(lambda :print("回车被按"))


#需求，按回车将焦点定位到下面的输入框
# lineEdit2  = QLineEdit(window)
# lineEdit2.move(100,150)
# lineEdit.returnPressed.connect(lambda :lineEdit2.setFocus())

#4,editingFinished 信号
# lineEdit.editingFinished.connect(lambda :print("结束编辑")) #它和按下回车键有些区别

#5，cursorPositionChanged  信号
# lineEdit.cursorPositionChanged.connect(lambda oldPos,newPos :print(oldPos,newPos))

#6，selectionChanged 信号
lineEdit.selectionChanged.connect(lambda :print("选中文本发生变化",lineEdit.selectedText()))

############################信号###############################

btn = QPushButton(window)
btn.setText("按钮")
btn.move(0,300)
# btn.clicked.connect(btn_clicked_slot)

#展示控件
window.show()

#3,进入消息循环
sys.exit(app.exec_())