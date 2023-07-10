from PyQt5.Qt import * #刚开始学习可以这样一下导入
import sys
#1,创建app
app  = QApplication(sys.argv)

#2，控件的操作：
#创建控件
window = QWidget()

#设置控件
window.setWindowTitle("交互状态")
window.resize(500,500)

window2 = QWidget()
window2.show()

window.show()

window2.raise_()  #将window2 提到最外层  ，但是它仍然不是出于活跃的状态

print(window2.isActiveWindow())  # False
print(window.isActiveWindow())   # True


#3,进入消息循环
sys.exit(app.exec_())