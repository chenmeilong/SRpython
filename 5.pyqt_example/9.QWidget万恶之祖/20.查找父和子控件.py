from PyQt5.Qt import * #刚开始学习可以这样一下导入
import sys
#1,创建app
app  = QApplication(sys.argv)
#2，控件的操作：
#创建控件
window = QWidget()

#设置控件
window.setWindowTitle("父子关系的学习")
window.resize(500,500)

label1 = QLabel(window)
        # label1.setParent(window)  也可以设置父亲
label1.setText("标签1")

label2 = QLabel(window)
label2.setText("标签2")
label2.move(100,100)

label3 = QLabel(window)
label3.setText("标签3")
label3.move(200,200)

print(window.childAt(101,105))  # 窗口 window 查看101,105 处是否有控件   输出：<PyQt5.QtWidgets.QLabel object at 0x0000021076265A68>

print(window.childAt(300,300))  # 窗口 window 查看300,300 处是否有控件   输出：None
print(window.childrenRect())  # 查看所有子控件的矩形区域  输出：PyQt5.QtCore.QRect(0, 0, 300, 230)
print(window.parentWidget())  # 查看父控件

#展示控件
window.show()

#3,进入消息循环
sys.exit(app.exec_())