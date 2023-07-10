from PyQt5.Qt import * #刚开始学习可以这样一下导入
import sys
#1,创建app
app  = QApplication(sys.argv)


#2，控件的操作：
#创建控件
window = QWidget()


#设置控件
window.setWindowTitle("模拟用户点击按钮")
window.resize(500,500)


btn = QPushButton(window)
btn.setText("按钮")
btn.move(200,200)
btn.pressed.connect(lambda :print("按钮鼠标被按下了"))

btn.released.connect(lambda :print("按钮鼠标被释放"))

btn.setCheckable(True)
btn.clicked.connect(lambda  arg:print("按钮被点击",arg))      #传参



###########################################################
btn.toggled.connect(lambda arg :print("按钮选中状态发生改变",arg))
#此时是不会触发它的，因为该按钮默认是不能被选中的。
btn.setCheckable(True)
###########################################################



#展示控件
window.show()

#3,进入消息循环
sys.exit(app.exec_())