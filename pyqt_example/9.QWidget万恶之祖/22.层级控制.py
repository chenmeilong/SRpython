from PyQt5.Qt import * #刚开始学习可以这样一下导入
import sys
#点谁 谁的层级就提高
class MyLabel(QLabel):
    def mousePressEvent(self, QMouseEvent):
        self.raise_()

#1,创建app
app  = QApplication(sys.argv)
#2，控件的操作：
#创建控件
window = QWidget()


#设置控件
window.setWindowTitle("层级控制")
window.resize(500,500)


label1 = MyLabel(window)
label1.setText("标签1")
label1.resize(200,200)
label1.setStyleSheet("background-color:red;")

label2 = MyLabel(window)
label2.setText("标签2")
label2.resize(200,200)
label2.setStyleSheet("background-color:green;")
label2.move(100,100)

#法一
# label1.raise_()  # 让label1 升到最高

#法二
# label2.lower()   # 将label2 降到最下层

#法三
label2.stackUnder(label1)   # 将label2 放在label1 的下面

#展示控件
window.show()

#3,进入消息循环
sys.exit(app.exec_())