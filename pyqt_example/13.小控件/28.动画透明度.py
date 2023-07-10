from PyQt5.Qt import * #刚开始学习可以这样一下导入
import sys

class Window(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("动画的学习")
        self.resize(400,400)
        self.set_ui()


    def set_ui(self):
        btn = QPushButton("按钮",self)
        btn.resize(200,200)
        btn.move(100,100)
        btn.setStyleSheet("background-color:cyan;")

        #1, 创建一个动画对象 ，并且设置目标属性
        animation = QPropertyAnimation(self,b"windowOpacity",self)  #可一起写，对象，属性
        # animation.setTargetObject(btn)  #对 btn 做动画
        # animation.setPropertyName(b"pos")  #对btn 的 pos 属性做动画


        #2，设置属性值  包括 开始值 （插值） 结束值
        # animation.setStartValue(1)
        # animation.setEndValue(0.5)


        #2，设置属性值  包括 开始值 （插值） 结束值
        animation.setStartValue(1)
        animation.setKeyValueAt(0.5,0.5)  #在动画时长的中间要变为 0.5
        animation.setEndValue(1)

        #3，动画时长
        animation.setDuration(3000)   #3s

        #动画方向设置一定要在 启动动画之前   动画方向
        # animation.setDirection(QAbstractAnimation.Backward)

        #4，启动动画
        animation.start()

        # 5，循环操作
        animation.setLoopCount(3)  # 循环三遍

        print("总时长", animation.totalDuration(), "单次时长", animation.duration())
        # btn.clicked.connect(lambda: print("当前时长", animation.currentLoopTime(), "当前循环内的时长", animation.currentTime()))

        #动画启动停止
        self.flag = True  #设置 标识
        def btn_clicked_slot():
            if self.flag :
                animation.pause()   #stop() 是不可恢复的   pause() 是可以恢复的
                self.flag  = False
            else:
                animation.resume()
                self.flag = True

        btn.clicked.connect(btn_clicked_slot)

if __name__ == '__main__':
    app =QApplication(sys.argv)

    window = Window()
    window.show()

    sys.exit(app.exec_())