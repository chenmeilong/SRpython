

from PyQt5.Qt import * #刚开始学习可以这样一下导入
import sys
#1,创建app
app  = QApplication(sys.argv)


#创建控件
window = QWidget()

window.move(100,100)
# window.resize(200,200)
window.setFixedSize(500,500)  # 创建窗口固定大小

print(window.geometry()) #输出： PyQt5.QtCore.QRect(100, 100, 200, 200)

#输出用户区域
window.show()
print(window.geometry())  #输出： PyQt5.QtCore.QRect(101, 131, 200, 200)

#3,进入消息循环
sys.exit(app.exec_())