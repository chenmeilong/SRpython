from PyQt5.Qt import * #刚开始学习可以这样一下导入
import sys
#1,创建app
app  = QApplication(sys.argv)


#2，控件的操作：

window = QMainWindow()  #组合控件
window.statusBar()

#设置控件
window.setWindowTitle("信息提示")
window.resize(500,500)


label = QLabel(window)
label.setText("hello world")
label.setToolTip("这是个标签")  #将鼠标停在上面的时候，在旁边会有提示
print(label.toolTip())   #获取工具提示
label.setToolTipDuration(2000)  # 显示2s   也可以默认5s

#展示控件
window.show()

#3,进入消息循环
sys.exit(app.exec_())