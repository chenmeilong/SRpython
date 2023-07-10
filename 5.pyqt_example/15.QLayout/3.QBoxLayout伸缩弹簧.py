from PyQt5.Qt import * #刚开始学习可以这样一下导入
import sys

class Window(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("QBoxLayout 的学习")
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
        boxLayout = QBoxLayout(QBoxLayout.TopToBottom)

        boxLayout.addWidget(label1)
        boxLayout.addStretch()  #加了个弹簧，其他控件是 建议的宽度
        boxLayout.addWidget(label2)
        boxLayout.addStretch()  #加了个弹簧，其他控件是 建议的宽度
        boxLayout.addWidget(label3)

        #设置伸缩因子  子控件
        boxLayout.setStretchFactor(label2,1)   #让label2可以自由拉伸
        #设置伸缩因子  子布局
        mylayout = QBoxLayout(QBoxLayout.LeftToRight)
        label4= QLabel("标签4")
        label4.setStyleSheet("background-color:cyan;")
        label5= QLabel("标签5")
        label5.setStyleSheet("background-color:blue;")

        mylayout.addWidget(label4)
        mylayout.addWidget(label5)

        boxLayout.addLayout(mylayout)
        #给子布局添加伸缩因子
        boxLayout.setStretchFactor(mylayout,1)

        self.setLayout(boxLayout)

        return None


if __name__ == '__main__':
    app =QApplication(sys.argv)

    window = Window()
    window.show()

    sys.exit(app.exec_())