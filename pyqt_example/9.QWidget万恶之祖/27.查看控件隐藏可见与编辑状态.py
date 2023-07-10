from PyQt5.Qt import * #刚开始学习可以这样一下导入
import sys
#1,创建app
app  = QApplication(sys.argv)

#2，控件的操作：
#创建控件
window = QWidget()

#设置控件
window.setWindowTitle("交互状态[*]")
window.resize(500,500)
window.setWindowModified(True)  # 此时*  就会被显示了，但是[] 不会被显示的
print(window.isWindowModified())  # 查看是否被被编辑的状态
btn = QPushButton(window)
btn.move(100,100)
btn.setText("按钮")

print(btn.isHidden())  #是否被设置隐藏
print(btn.isVisible()) #到此是否可见

window.show()        #此时：   输出：  False / True

print(btn.isHidden())  #是否被设置隐藏
print(btn.isVisible()) #到此是否可见

#展示控件

#3,进入消息循环
sys.exit(app.exec_())
