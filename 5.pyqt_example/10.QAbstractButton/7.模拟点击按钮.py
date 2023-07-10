from PyQt5.Qt import * #刚开始学习可以这样一下导入
import sys
#1,创建app
app  = QApplication(sys.argv)
#2，控件的操作：
#创建控件
window = QWidget()
#设置控件
window.setWindowTitle("模拟用户点击按钮")
window.resize(500,500)
btn = QPushButton(window)
btn.setText("按钮")
btn.move(200,200)
btn.pressed.connect(lambda :print("按钮被点击了"))
############################模拟用户点击###############################
# btn.click()            #点击按钮
# btn.click()

btn.animateClick(2000)  # 长按2s ，让按钮被选中2s,之后再消失，按钮有生命2s
                                # 2s的动画效果
############################模拟用户点击###############################
#展示控件
window.show()
#3,进入消息循环
sys.exit(app.exec_())