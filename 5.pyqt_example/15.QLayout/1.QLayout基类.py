from PyQt5.Qt import * #刚开始学习可以这样一下导入
import sys

class Window(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("QLayout 的学习")
        self.resize(400,400)
        self.set_ui()


    def set_ui(self):
        label1= QLabel("标签1")
        label1.setStyleSheet("background-color:red;")
        label2= QLabel("标签2")
        label2.setStyleSheet("background-color:green;")
        label3= QLabel("标签3")
        label3.setStyleSheet("background-color:yellow;")


        #
        boxLayout = QBoxLayout(QBoxLayout.BottomToTop)   #从下到上垂直

        boxLayout.addWidget(label1)
        boxLayout.addWidget(label2)
        boxLayout.addWidget(label3)

        boxLayout.setSpacing(60)      #小控件之间的间距
        boxLayout.setContentsMargins(0,0,0,0)

        #设置能用性
        # boxLayout.setEnabled(False)


        #替换子控件
        label4 = QLabel("标签4")
        label4.setStyleSheet("background-color:orange;")
        boxLayout.replaceWidget(label2,label4)
        #替换后一般要隐藏要替换的
        label2.hide()
        #删除 label2  的特殊方式，就是让它的父控件为None
        label2.setParent(None)  #这也释放了label2 ,如果要验证它，可以通过信号


        #添加子布局 (布局的嵌套)
        label5= QLabel("标签5")
        label5.setStyleSheet("background-color:pink;")
        label6= QLabel("标签6")
        label6.setStyleSheet("background-color:blue;")
        label7= QLabel("标签7")
        label7.setStyleSheet("background-color:cyan;")

        h_layout = QBoxLayout(QBoxLayout.LeftToRight) #水平布局  左到右

        h_layout.addWidget(label5)
        h_layout.addWidget(label6)
        h_layout.addWidget(label7)

        boxLayout.addLayout(h_layout)
        #添加子布局 (布局的嵌套)

        self.setLayout(boxLayout)

        timer = QTimer(self)
        timer.timeout.connect(lambda :label3.setText(label3.text()+"Hello\n"))
        timer.start(1000)  #1s


if __name__ == '__main__':
    app =QApplication(sys.argv)

    window = Window()
    window.show()

    sys.exit(app.exec_())