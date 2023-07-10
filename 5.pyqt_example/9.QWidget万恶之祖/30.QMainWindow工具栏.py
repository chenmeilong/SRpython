from PyQt5.Qt import * #刚开始学习可以这样一下导入
import sys
#1,创建app
app  = QApplication(sys.argv)


#2，控件的操作：

window = QMainWindow()  #组合控件  从外到内依次是菜单栏、状态栏、工具栏、停靠窗口、中心窗口是 qwidget的组合控件
window.statusBar()

#设置控件
window.setWindowTitle("信息提示")
window.resize(500,500)

#当鼠标停留在窗口控件之后，在状态栏展示提示信息，但是前提是要有状态栏
window.setStatusTip("这是窗口")
print(window.statusTip()) #获取提示信息
label = QLabel(window)
label.setText("hello world") 
label.setStatusTip("这是标签")

#展示控件
window.show()

#3,进入消息循环
sys.exit(app.exec_())