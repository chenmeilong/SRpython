from PyQt5.Qt import *  # 刚开始学习可以这样一下导入
import sys


class Window(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("QObject的学习")
        self.resize(400, 400)
        self.set_ui()
    def set_ui(self):
        self.QObject_test()
    def QObject_test(self):
        with open("QObject.qss", "r") as f:
            qApp.setStyleSheet(f.read())

        label = QLabel(self)
        label.setText("hello 世界")
        label.setObjectName("notice")

        label2 = QLabel(self)
        label2.setObjectName("notice")
        label2.setProperty("notice_level", "warning")
        label2.setText("你好，world")
        label2.move(100, 100)

        label3 = QLabel(self)
        label3.setText("你好，world")
        label3.setObjectName("notice")
        label3.setProperty("notice_level","error")
        label3.move(200, 200)


        box1 = QWidget(self)
        box2 = QWidget(self)

        # box1.setStyleSheet("background-color:orange;")
        # box2.setStyleSheet("background-color:cyan;")
        box1.setStyleSheet("QPushButton {background-color:orange;}")   #过滤
        # 给box1 加上选择器

        # box1
        label1 = QLabel("标签1", box1)
        label1.move(50, 50)
        btn1 = QPushButton("按钮1", box1)
        btn1.move(100, 100)

        # box2
        label2 = QLabel("标签1", box2)
        label2.move(50, 50)
        btn2 = QPushButton("按钮1", box2)
        btn2.move(100, 100)

        ###########################################################
        # 此时box1 和box2 中的控件 也默认跟随它们设置的颜色
        ###########################################################

        v_layout = QVBoxLayout()
        self.setLayout(v_layout)

        v_layout.addWidget(box1)
        v_layout.addWidget(box2)


if __name__ == '__main__':
    app = QApplication(sys.argv)

    # app.setStyleSheet("QPushButton {background-color:orange;}")
    #给整个app 加颜色     就是里面所有的元素添加颜色

    window = Window()
    window.show()

    sys.exit(app.exec_())