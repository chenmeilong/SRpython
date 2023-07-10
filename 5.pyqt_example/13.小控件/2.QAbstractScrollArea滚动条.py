from PyQt5.Qt import * #刚开始学习可以这样一下导入
import sys

#QTextEdit和QPlainTextEdit  都是继承它的
#1,创建app
app  = QApplication(sys.argv)


#2，控件的操作：
#创建控件
window = QWidget()


#设置控件
window.setWindowTitle("QAbstractScrollArea 的学习（通过QTextEdit) ")
window.resize(500,500)

textEdit  = QTextEdit("Hello python!",window)

textEdit.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
textEdit.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOn)

############################角落控件###############################
btn= QPushButton(window)
btn.setIcon(QIcon("icon.ico"))
textEdit.setCornerWidget(btn)
############################角落控件###############################


#展示控件
window.show()

#3,进入消息循环
sys.exit(app.exec_())