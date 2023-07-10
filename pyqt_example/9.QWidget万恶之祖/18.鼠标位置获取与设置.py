from PyQt5.Qt import * #刚开始学习可以这样一下导入
import sys
#1,创建app
app  = QApplication(sys.argv)
#2，控件的操作：
#创建控件
window = QWidget()
#设置控件
window.setWindowTitle("鼠标操作")
window.resize(500,500)

current_cursor = window.cursor()  #获取鼠标对象

print(current_cursor.pos())  #获取鼠标的位置   PyQt5.QtCore.QPoint(748, 260)# 它是相对于整个电脑屏幕的
#为了验证这一点，我们通过下的验证

current_cursor.setPos(0,0)  # 这时鼠标的位置在 屏幕的左上角

#展示控件
window.show()
#3,进入消息循环
sys.exit(app.exec_())