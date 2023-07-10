from PyQt5.Qt import * #刚开始学习可以这样一下导入
import sys

class Window(QWidget):
    def mousePressEvent(self, event):
        # self.focusNextChild()  #在子控件中切换焦点
        # self.focusPreviousChild()  #反序
        self.focusNextPrevChild(True)  #True 是前面的Next false 是后面的Prev
        print(self.focusWidget())  #点击时获取它的子控件中获取焦点的那个

#1,创建app
app  = QApplication(sys.argv)


#2，控件的操作：
#创建控件
window = Window()


#设置控件
window.setWindowTitle("焦点控制")
window.resize(500,500)

lineEdit1 = QLineEdit(window)
lineEdit1.move(50,50)

lineEdit2 = QLineEdit(window)
lineEdit2.move(100,100)
lineEdit2.setFocus()  #先让第二个获取焦点

lineEdit3 = QLineEdit(window)
lineEdit3.move(150,150)


Window.setTabOrder(lineEdit1,lineEdit3)
Window.setTabOrder(lineEdit3,lineEdit2)
#tab 切换  1  3  2


#展示控件
window.show()



#3,进入消息循环
sys.exit(app.exec_())