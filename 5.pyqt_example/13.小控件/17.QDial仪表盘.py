from PyQt5.Qt import * #刚开始学习可以这样一下导入
import sys

class Window(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("QDial的学习")
        self.resize(400,400)
        self.set_ui()


    def set_ui(self):
        dial = QDial(self)

        # dial.valueChanged.connect(lambda val:print(val))
        dial.setRange(0,200)

        #显示刻度
        dial.setNotchesVisible(True)

        #改变步长
        dial.setPageStep(5)

        #让刻度包裹 整个圆
        dial.setWrapping(True)

        #刻度之间的间隔
        dial.setNotchTarget(10)

        ############################改变字体###############################
        label = QLabel(self)
        label.move(100,200)
        label.setText("Life is short,I learn Python!")

        def test_slot(val):
            label.setStyleSheet("font-size:{}px".format(val))
            label.adjustSize()
        dial.valueChanged.connect(test_slot)
        ############################改变字体###############################

if __name__ == '__main__':
    app =QApplication(sys.argv)

    window = Window()
    window.show()

    sys.exit(app.exec_())