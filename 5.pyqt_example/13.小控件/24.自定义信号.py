from PyQt5.Qt import * #刚开始学习可以这样一下导入
import sys

class Btn(QPushButton):
    right_clicked = pyqtSignal([str],[int],[int,int,str])  #中括号的意思是重载，有可能发射 str 也有可能int

    def mousePressEvent(self,event):
        super().mousePressEvent(event)

        if event.button() == Qt.RightButton:  #这里解决了什么时候发射信号
            # print("应该发射右击信号")
            self.right_clicked.emit(self.text())   #这里解决了如何将自定义信号发射出去
            self.right_clicked[int,int,str].emit(1,1,self.text())   #这里解决了如何将自定义信号发射出去
            self.right_clicked[int].emit(1)   #这里解决了如何将自定义信号发射出去

class Window(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("信号的学习")
        self.resize(400,400)
        self.set_ui()
    def set_ui(self):
        btn = Btn("按钮",self)
        btn.right_clicked.connect(lambda arg:print("右键被点击了",arg))
        # btn.right_clicked[int,int,str].connect(lambda v1,v2,arg :print("右键被点击了",v1,v2,arg))
        # btn.right_clicked[int].connect(lambda val:print(val))

if __name__ == '__main__':
    app =QApplication(sys.argv)

    window = Window()
    window.show()

    sys.exit(app.exec_())