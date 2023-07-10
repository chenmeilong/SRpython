from PyQt5.Qt import * #刚开始学习可以这样一下导入
import sys

class Window(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("QSpinBox 的学习")
        self.resize(400,400)
        self.set_ui()


    def set_ui(self):
        self.spinBox = QSpinBox(self)
        self.spinBox.resize(100,30)
        self.spinBox.move(100,100)


        self.test()

    def test(self):
        # self.spinBox.setDisplayIntegerBase(2)  # 2进制
        self.spinBox.setRange(-100,200)  #注：它这个两边都是可以取到的   设置最大最小值
        self.spinBox.setAccelerated(True)  #  #设置加速

        self.spinBox.setSingleStep(10)    #设置步长
        print("步长：",self.spinBox.singleStep())


        print(self.spinBox.wrapping())
        self.spinBox.setWrapping(True)  #数值循环



if __name__ == '__main__':
    app =QApplication(sys.argv)

    window = Window()
    window.show()

    sys.exit(app.exec_())