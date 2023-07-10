

from PyQt5.Qt import *  # 刚开始学习可以这样一下导入
import sys
app = QApplication(sys.argv)

win1 = QWidget()
win1.setStyleSheet("background-color:red;")
win1.show()

win2 = QWidget()
win2.setStyleSheet("background-color:green;")
win2.resize(100,100)
win2.setParent(win1)  #将win2 放到win1 窗口中


# win2.gridLayout = QGridLayout()
# label1 = QLabel("标签1")
# label1.setStyleSheet("background-color:red;")
# win2.gridLayout.addWidget(label1, 0, 0)

btn = QPushButton(win2)
btn.setText("xxx")


# win2.setStyleSheet("background-color:black;")  #依然可以这样修改win2的属性

win2.show()

sys.exit(app.exec_())

