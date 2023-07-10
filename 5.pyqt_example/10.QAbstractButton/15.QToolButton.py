from PyQt5.Qt import * #刚开始学习可以这样一下导入
import sys
#1,创建app
app  = QApplication(sys.argv)


#2，控件的操作：
#创建控件
window = QWidget()


#设置控件
window.setWindowTitle("QCommandLinkButton功能作用")
window.resize(500,500)

bool_btn = QToolButton(window)
bool_btn.setText("工具按钮")



############################当有图标显示时候，文本就不会显示了###############################
#它和QPushButton 的不同，就是在于它一般只显示图标，不显示文本，而
#QpushButton 既显示文本又显示图标。

bool_btn.setIcon(QIcon("icon.ico"))
bool_btn.setIconSize(QSize(50,50))
############################自动提升效果###############################
bool_btn.setAutoRaise(True)
#当鼠标放上去的时候，会有自动提升的效果



#注：图标大小也可修改！
############################当有图标显示时候，文本就不会显示了###############################
bool_btn.setToolTip("这是一个新建按钮")         #可以设置提示

# bool_btn.setToolButtonStyle(Qt.ToolButtonTextUnderIcon)  #这样就能图标文字一起显示

#展示控件
window.show()

#3,进入消息循环
sys.exit(app.exec_())