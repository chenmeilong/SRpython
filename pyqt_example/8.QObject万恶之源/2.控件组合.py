from PyQt5.Qt import * #刚开始学习可以这样一下导入
import sys

app  = QApplication(sys.argv)  #sys.argv 是个列表,它的0号元素是文件名


#2，控件的操作：
#创建控件

window = QWidget()

label = QLabel(window)  # 将label 控件添加到window 控件上
#设置控件
window.setWindowTitle("社会")
window.resize(400,400)
label.setText("hello world")
label.move(200,200)

#展示控件
window.show()
sys.exit(app.exec_())