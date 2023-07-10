from PyQt5.Qt import * #刚开始学习可以这样一下导入
import sys
#1,创建app
app  = QApplication(sys.argv)


#创建控件
window = QWidget()



window.show()  #显示子控件时是这样的，它必须要在顶层窗口的前面，
                #因为当顶层窗口显示时，它会遍历它身上的所有的子控件
window.resize(500,500)
window.move(300,300)

w = QWidget(window)
w.resize(100,100)
w.setStyleSheet("background-color:red;")
w.show() # 自己手动展示          #没有这行手动显示则不能显示

#3,进入消息循环
sys.exit(app.exec_())