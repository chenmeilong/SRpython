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



        btn = QPushButton(self)
        btn.setText("按钮")
        btn.move(0,300)
        btn.clicked.connect(lambda :self.test())

    def test(self):
        ############################最小值特殊文本###############################

        #前缀
        self.spinBox.setRange(0,6)
        self.spinBox.setPrefix("周")
        #这时在最小值的时候就是周0 了，这样不行的 ，要用到下个方法：下面看
        self.spinBox.setSpecialValueText("周日")


        print(self.spinBox.value())    #此时并没有获取前缀，只是 数值
        print(self.spinBox.text())
        print(self.spinBox.lineEdit().text())
        ############################最小值特殊文本###############################




if __name__ == '__main__':
    app =QApplication(sys.argv)

    window = Window()
    window.show()

    sys.exit(app.exec_())