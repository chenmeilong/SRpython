#QSpinBox 里的基本一致，只是数值的类型不同而已。

from PyQt5.Qt import * #刚开始学习可以这样一下导入
import sys

class Window(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("QDoubleSpinBox的学习")
        self.resize(400,400)
        self.set_ui()


    def set_ui(self):
        doubleSpinBox = QDoubleSpinBox(self)
        doubleSpinBox.resize(100,30)
        doubleSpinBox.move(100,30)

        doubleSpinBox.setRange(1.0,2.0)
        doubleSpinBox.setSingleStep(0.5)
        doubleSpinBox.setSuffix("倍速")  #设置后缀

        doubleSpinBox.setSpecialValueText("正常")

        doubleSpinBox.setWrapping(True)

        #设置小数位数
        doubleSpinBox.setDecimals(1)  #保留一位小数
        print(doubleSpinBox.decimals())



if __name__ == '__main__':
    app =QApplication(sys.argv)

    window = Window()
    window.show()

    sys.exit(app.exec_())