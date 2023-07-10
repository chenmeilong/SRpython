from PyQt5.Qt import * #刚开始学习可以这样一下导入
import sys

class Window(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("QAbstractSlider的学习")
        self.resize(400,400)
        self.set_ui()


    def set_ui(self):
        slider  = QSlider(self)
        self.slider = slider


        #设置最大值 和最小值
        slider.setMaximum(100)
        slider.setMinimum(66)

        #通过 valueChanged 来查看数值范围
        slider.valueChanged.connect(lambda val:self.label.setText(str(val)))
        #可以通过追踪设置改变 这个事件的触发条件

        #步长设置
        slider.setSingleStep(5)  #步长指的是  上下键的步长
        slider.setPageStep(10)  # 按翻页的步长

        #追踪设置
        print(slider.hasTracking())
        slider.setTracking(False)  # 这时，val 值就不会随着滑块变动而变动了

        #滑块位置
        slider.setSliderPosition(88)

        # 倒立外观
        # slider.setInvertedAppearance(True)  # 最小的在最上  最大在最下面
        # slider.setInvertedControls(True)  #  将键盘控制也改了

        # 改为水平
        slider.setOrientation(Qt.Horizontal)

        ############################信号###############################
        #1
        # slider.sliderMoved.connect(lambda val:print(val))  #滑块鼠标拖动信号
        #2
        # slider.actionTriggered.connect(lambda val:print(val))
        #3
        # slider.rangeChanged.connect(lambda min,max:print(min,max))   #滑块范围改变信号
        ############################信号###############################


        btn = QPushButton(self)
        btn.setText("按钮")
        btn.move(0, 300)
        btn.clicked.connect(self.btn_clicked_slot)

        self.test()

    def test(self):
        label = QLabel(self)
        label.setText(str(self.slider.value()))  #填充当前滑块的值
        label.move(200,200)
        label.resize(100, 30)
        self.label = label

    def btn_clicked_slot(self):
        print("当前的数值：", self.slider.value())
        self.slider.setMaximum(99)

if __name__ == '__main__':
    app =QApplication(sys.argv)

    window = Window()
    window.show()

    sys.exit(app.exec_())
