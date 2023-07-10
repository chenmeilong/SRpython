from PyQt5.Qt import * #刚开始学习可以这样一下导入
import sys
#1,创建app
app  = QApplication(sys.argv)


#2，控件的操作：
#创建控件
window = QWidget()


#设置控件
window.setWindowTitle("最小尺寸和最大尺寸的限定")

window.setMinimumSize(200,200)  #当然也可单独限制高和宽
window.setMaximumSize(600,600)

window.resize(800,800)        #不可修改尺寸


#展示控件
window.show()

#3,进入消息循环
sys.exit(app.exec_())