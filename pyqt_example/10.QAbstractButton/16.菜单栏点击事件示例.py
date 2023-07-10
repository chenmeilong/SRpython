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

tool_btn = QToolButton(window)

tool_btn.setIcon(QIcon("icon.ico"))
tool_btn.setIconSize(QSize(50,50))

tool_btn.setAutoRaise(True)
#当鼠标放上去的时候，会有自动提升的效果

############################给工具按钮设置菜单###############################

# btn = QPushButton(window)
# btn.setFlat(True)
# btn.move(100,100)
# btn.setText("菜单")

menu = QMenu(tool_btn)

sub_menu = QMenu(menu)
sub_menu.setTitle("子菜单")
relative_action  = QAction("最近打开")
sub_menu.addAction(relative_action)


new_action = QAction("新建")
new_action.setData("new")
menu.addAction(new_action)

open_action = QAction("打开")
open_action.setData("open")
menu.addAction(open_action)

menu.addSeparator()
menu.addMenu(sub_menu)


tool_btn.setMenu(menu)
tool_btn.setPopupMode(QToolButton.InstantPopup) # #设置成整体点击弹出
############################给工具按钮设置菜单###############################

##########################通过给action 绑定数据来区分不同action#################################
def tool_btn_triggered_slot(action):
    if action.data()  == "new":
        print("你点击的是新建")
    elif action.data() == "open":
        print("你点击的是打开")


tool_btn.triggered.connect(tool_btn_triggered_slot)

##########################通过给action 绑定数据来区分不同action#################################


#展示控件
window.show()

#3,进入消息循环
sys.exit(app.exec_())