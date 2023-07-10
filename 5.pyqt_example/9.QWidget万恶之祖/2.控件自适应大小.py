from PyQt5.Qt import * #刚开始学习可以这样一下导入
import sys
#1,创建app
app  = QApplication(sys.argv)
#创建控件
window = QWidget()

window.move(200,200)
window.resize(500,500)

label = QLabel(window)
label.setText("hello world")
label.move(100,100)
label.setStyleSheet("background-color:cyan;")  #为了更清晰看到具体的区域

def addContent():
    new_content = label.text() +"hello world"
    label.setText(new_content)
    # label.resize(label.width()+100,label.height())  笨方法
    label.adjustSize()      #自适应大小


btn = QPushButton(window)
btn.setText("增加内容")
btn.move(200,200)
btn.clicked.connect(addContent)

window.show()
#3,进入消息循环
sys.exit(app.exec_())
# 增加内容 自适应大小