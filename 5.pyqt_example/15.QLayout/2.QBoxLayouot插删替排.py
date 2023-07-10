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
        boxLayout = QBoxLayout(QBoxLayout.BottomToTop)

        # boxLayout.setDirection(QBoxLayout.TopToBottom)  #改变排列的方式

        boxLayout.addWidget(label1)

        #添加空白  （空白大小不会随着缩放而变化）
        # boxLayout.addSpacing(100)  # setspacing 是给每一个设置

        boxLayout.addWidget(label2)

        # 添加伸缩 因子  （弹簧）
        # boxLayout.addStretch(2)  #这里需要注意的是，当空间被压缩的很小的时候，压缩因子就失效了

        boxLayout.addWidget(label3)

        #添加元素
        label4= QLabel("标签4")
        label4.setStyleSheet("background-color:cyan;")
        label5= QLabel("标签5")
        label5.setStyleSheet("background-color:blue;")
        #添加widget
        # boxLayout.insertWidget(1,label4)
        #添加layout
        mylayout = QBoxLayout(QBoxLayout.LeftToRight)
        mylayout.addWidget(label4)
        mylayout.addWidget(label5)
        boxLayout.insertLayout(2,mylayout)

        #移除label1
        # boxLayout.removeWidget(label1)
        # label1.hide()


        self.setLayout(boxLayout)


if __name__ == '__main__':
    app =QApplication(sys.argv)

    window = Window()
    window.show()

    sys.exit(app.exec_())