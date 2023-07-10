from PyQt5.Qt import * #刚开始学习可以这样一下导入
import sys

class Window(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("QStackedLayout的学习")
        self.resize(400,400)
        self.set_ui()


    def set_ui(self):
        stackedLayout = QStackedLayout()

        self.setLayout(stackedLayout)  #一般先设置它

        label1= QLabel("标签1")
        label1.setStyleSheet("background-color:red;")
        label2= QLabel("标签2")
        label2.setStyleSheet("background-color:green;")
        label3= QLabel("标签3")
        label3.setStyleSheet("background-color:yellow;")
        label4= QLabel("标签4")
        label4.setStyleSheet("background-color:cyan;")
        label5= QLabel("标签5")
        label5.setStyleSheet("background-color:blue;")

        v_layout = QVBoxLayout()
        v_layout.addWidget(label4)
        v_layout.addWidget(label5)

        stackedLayout.addWidget(label1)
        stackedLayout.addWidget(label2)
        stackedLayout.addWidget(label3)




        #展示模式修改
        # label1.hide()  #此时后面的也不会显示出来
        #
        # stackedLayout.setStackingMode(QStackedLayout.StackAll)
        # label1.hide()  # 此时，如果label1 不显示，它后面的也会显示
        # stackedLayout.setStackingMode(QStackedLayout.StackAll)  #所有小控件可见，当前控件在最前面
        # label1.setFixedSize(100,100)


        #轮流显示 各个标签
        timer = QTimer(self)
        def timeout_slot():
            stackedLayout.setCurrentIndex((stackedLayout.currentIndex()+1)%stackedLayout.count())
        timer.timeout.connect(timeout_slot)
        timer.start(500)

        #删除控件
        # stackedLayout.removeWidget(label1)  # 此时，后面也会自动显示

        #信号
        stackedLayout.currentChanged.connect(lambda val:print(val))




if __name__ == '__main__':
    app =QApplication(sys.argv)

    window = Window()
    window.show()

    sys.exit(app.exec_())