from PyQt5.Qt import * #刚开始学习可以这样一下导入
import sys

class Window(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("动画组的学习")
        self.resize(800,800)
        self.set_ui()


    def set_ui(self):
        red_btn = QPushButton("红色按钮",self)
        green_btn = QPushButton("绿色按钮",self)

        red_btn.resize(100,100)

        green_btn.resize(100,100)
        green_btn.move(150,150)

        red_btn.setStyleSheet("background-color:red;")
        green_btn.setStyleSheet("background-color:green;")

        #动画设置
        ###########################################################
        animation_green = QPropertyAnimation(green_btn,b"pos",self)

        animation_green.setKeyValueAt(0,QPoint(150,150))
        animation_green.setKeyValueAt(0.25,QPoint(550,150))
        animation_green.setKeyValueAt(0.5,QPoint(550,550))
        animation_green.setKeyValueAt(0.75,QPoint(150,550))
        animation_green.setKeyValueAt(1,QPoint(150,150))

        animation_green.setDuration(5000)
        animation_green.setLoopCount(3)

        # animation_green.start()  # 动画不是阻塞的， 这一行不会阻塞

        ###########################################################
        animation_red = QPropertyAnimation(red_btn,b"pos",self)
        animation_red.setKeyValueAt(0,QPoint(0,0))
        animation_red.setKeyValueAt(0.25,QPoint(0,700))
        animation_red.setKeyValueAt(0.5,QPoint(700,700))
        animation_red.setKeyValueAt(0.75,QPoint(700,0))
        animation_red.setKeyValueAt(1,QPoint(0,0))
        animation_red.setDuration(5000)
        animation_red.setLoopCount(3)
        # animation_red.start()


        #用动画组来管理上面两个动画
        animation_group  = QParallelAnimationGroup(self)
        animation_group.addAnimation(animation_red)
        animation_group.addAnimation(animation_green)

        animation_group.start()

        red_btn.clicked.connect(animation_group.pause)
        green_btn.clicked.connect(animation_group.resume)




if __name__ == '__main__':
    app =QApplication(sys.argv)

    window = Window()
    window.show()

    sys.exit(app.exec_())