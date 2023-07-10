from PyQt5.Qt import *  # 刚开始学习可以这样一下导入
import sys


class Window(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("QLCDNumber的学习")
        self.resize(400, 400)
        self.set_ui()

    def set_ui(self):
        # lcdNumber = QLCDNumber(self)
        lcdNumber = QLCDNumber(5, self)  # 5指的是 5 位
        lcdNumber.resize(300, 50)
        lcdNumber.move(100, 100)

        # 设置显示数值
        # lcdNumber.display("12345")
        # lcdNumber.display("osgabcdefhlpruy")
        # lcdNumber.display(": '")  #冒号，空格，单引号（°）


        #模式设置
        # lcdNumber.setMode(QLCDNumber.Bin) #二进制
        # lcdNumber.setMode(QLCDNumber.Oct) #八进制
        # lcdNumber.setMode(QLCDNumber.Hex) #十六进制


        #溢出检测
        lcdNumber.overflow.connect(lambda :print("数值溢出"))
        print(lcdNumber.checkOverflow(99))
        print(lcdNumber.checkOverflow(888888))

        btn = QPushButton(self)
        btn.setText("按钮")
        btn.move(0, 300)
        btn.clicked.connect(lambda: print(lcdNumber.value()))

        # 位数限制
        lcdNumber.setDigitCount(5)
        # lcdNumber.setNumDigits(2)

        # 展示数字
        # 注意的是，如果数字的位数大于给定的，那么 会显示0 ，而且会发出一个信号（溢出）
        lcdNumber.display(888888)
        # lcdNumber.display(-10)

        #分段样式  lcd的样式
        # lcdNumber.setSegmentStyle(QLCDNumber.Outline)
        lcdNumber.setSegmentStyle(QLCDNumber.Filled)
        # lcdNumber.setSegmentStyle(QLCDNumber.Flat)


if __name__ == '__main__':
    app = QApplication(sys.argv)

    window = Window()
    window.show()

    sys.exit(app.exec_())