from PyQt5.Qt import * #刚开始学习可以这样一下导入
import sys
#1,创建app
app  = QApplication(sys.argv)


#2，控件的操作：
#创建控件
window = QWidget()


#设置控件
window.setWindowTitle("QDialog的学习")
window.resize(500,500)

dialog = QDialog(window)  #对话框和 主窗口不关联
dialog.setWindowTitle("对话框")

btn = QPushButton(dialog)
btn.setText("btn1")
btn.move(20,20)
btn.clicked.connect(lambda :dialog.accept())  #接收

btn2 = QPushButton(dialog)
btn2.setText("btn2")
btn2.move(60,60)
btn2.clicked.connect(lambda :print(dialog.result()))  # 此时点 btn2 和 btn3 都不会关闭对话框了

btn3 = QPushButton(dialog)
btn3.setText("btn3")
btn3.move(100,100)
btn3.clicked.connect(lambda :dialog.setResult(888))


ret  = dialog.exec()
print(ret)


#展示控件
window.show()

#3,进入消息循环
sys.exit(app.exec_())