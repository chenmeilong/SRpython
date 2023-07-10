from PyQt5.Qt import * #刚开始学习可以这样一下导入
import sys
#1,创建app
app  = QApplication(sys.argv)

#2，控件的操作：
#创建控件
window = QWidget()

#设置控件
window.setWindowTitle("顶层窗口相关操作")

window.resize(500,500)

icon = QIcon("icon.png")
window.setWindowIcon(icon)  #设置图标
print(window.windowIcon())  #获取图标
print(window.windowTitle())  #获取标题

window.setWindowOpacity(0.9)  # 设置为半透明
print(window.windowOpacity())  # 获取不透明度


#法一
# window.setWindowState(Qt.WindowMinimized)  #最小化
# window.setWindowState(Qt.WindowMaximized)  #最大化  有标题栏
# window.setWindowState(Qt.WindowFullScreen)  #全屏  标题栏都没了 #这时要通过任务管理器结束进程

# window.show()

#法二 （常用）
#展示控件
# window.show()
window.showMaximized()  #展示最大，这时上面的window.show() 也可以不用要
# window.showFullScreen()
# window.showMinimized()

#3,进入消息循环
sys.exit(app.exec_())