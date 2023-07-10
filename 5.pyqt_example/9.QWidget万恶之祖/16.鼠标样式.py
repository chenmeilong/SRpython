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

window.setCursor(Qt.BusyCursor)   #转圈圈的样式

#展示控件
window.show()

#3,进入消息循环
sys.exit(app.exec_())