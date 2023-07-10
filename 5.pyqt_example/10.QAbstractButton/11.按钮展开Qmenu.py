from PyQt5.Qt import * #刚开始学习可以这样一下导入
import sys

#1,创建app
app  = QApplication(sys.argv)


#2，控件的操作：
#创建控件
window = QWidget()


#设置控件
window.setWindowTitle("QAbstactButton 有效区域")
window.resize(500,500)


btn = QPushButton(window)
btn.setText("xxx")
btn.setIcon(QIcon("icon.ico"))

############################菜单的设置###############################

menu = QMenu()

#菜单之行为动作 新建，打开
new_action = QAction(menu)  # 父控件是menu
new_action.setText("新建")
new_action.setIcon(QIcon("icon.ico"))
new_action.triggered.connect(lambda :print("新建文件"))

open_action = QAction(menu)  # 父控件是menu
open_action.setText("打开")
open_action.setIcon(QIcon("icon.ico"))
open_action.triggered.connect(lambda :print("打开文件"))

exit_action = QAction(menu)  # 父控件是menu
exit_action.setText("退出")
exit_action.setIcon(QIcon("icon.ico"))
exit_action.triggered.connect(lambda :print("退出"))

menu.addAction(new_action)
menu.addAction(open_action)
menu.addAction(exit_action)

menu.addSeparator()
#菜单之子菜单   最近打开

sub_menu = QMenu(menu)
sub_menu.setTitle("最近打开 ")  # 注意不是steText
sub_menu.setIcon(QIcon("icon.ico"))

file_action = QAction("Python gui 编程 PyQt5")
sub_menu.addAction(file_action)


menu.addMenu(sub_menu)


btn.setMenu(menu)

############################菜单的设置###############################



#展示控件
window.show()

###########################################################
btn.showMenu()  #此时，先展示的是菜单，它可以独自展示的，因为它直接继承的QWidget
###########################################################

#3,进入消息循环
sys.exit(app.exec_())