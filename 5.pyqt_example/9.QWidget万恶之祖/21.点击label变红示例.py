from PyQt5.Qt import * #刚开始学习可以这样一下导入
import sys

class Window(QWidget):
    def mousePressEvent(self, event):
        local_x = event.x()
        local_y = event.y()
        sub_widget = self.childAt(local_x,local_y)  #按位置找子控件
        print(sub_widget)
        if sub_widget:  # 排除sub_widget 是None
            sub_widget.setStyleSheet("background-color:red;")

#1,创建app
app  = QApplication(sys.argv)

#2，控件的操作：
#创建控件
window = Window()

#设置控件
window.setWindowTitle("父子关系案例")
window.resize(500,500)
for i in range(10):
    label = QLabel(window)
    label.setText("标签"+str(i))
    label.move(30*i ,30*i)
#展示控件
window.show()

#3,进入消息循环
sys.exit(app.exec_())