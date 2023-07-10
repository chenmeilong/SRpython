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
        animation = QPropertyAnimation(self)
        animation.setTargetObject(btn)  #对 btn 做动画
        animation.setPropertyName(b"pos")  #对btn 的 pos 属性做动画 常用geometry、pos、size、windowOpacity


        #2，设置属性值  包括 开始值 （插值） 结束值
        animation.setStartValue(QPoint(0,0))
        animation.setEndValue(QPoint(300,300))

        #3，动画时长
        animation.setDuration(3000)   #3s

        #4，启动动画
        animation.start()

if __name__ == '__main__':
    app =QApplication(sys.argv)

    window = Window()
    window.show()

    sys.exit(app.exec_())