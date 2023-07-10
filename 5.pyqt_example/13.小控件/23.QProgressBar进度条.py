from PyQt5.Qt import *  # 刚开始学习可以这样一下导入
import sys


class Window(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("QProgressBar的学习")
        self.resize(400, 400)
        self.set_ui()

    def set_ui(self):
        progressBar = QProgressBar(self)
        self.progressBar = progressBar
        print(progressBar.minimum())
        print(progressBar.maximum())
        # progressBar.setMaximum(200)
        # progressBar.setRange(0,200)

        progressBar.resize(400, 30)
        progressBar.setValue(50)

        #显示文字
        progressBar.setFormat("当前人数:{}/总人数%m".format(progressBar.value() - progressBar.minimum()))  #
        self.progressBar.setAlignment(Qt.AlignCenter)   #设置文字显示位置


        # 繁忙状态
        # progressBar.setRange(0,0 )

        btn = QPushButton(self)
        btn.setText("按钮")
        btn.move(0, 300)
        btn.clicked.connect(self.btn_clicked_slot)

    def btn_clicked_slot(self):
        # self.progressBar.reset()  #重置
        # self.progressBar.setInvertedAppearance(True)  #反转

        #文本操作
        # self.progressBar.setTextVisible(False)
        # print(self.progressBar.text())

        self.progressBar.resize(30,200)
        self.progressBar.setOrientation(Qt.Vertical)   #设置成竖直进度条
        print(self.progressBar.isVisible())  #True ，但是看不到

        timer = QTimer(self.progressBar)  #定时器归 进度条拥有
        def timer_func():
            # print("xxx")
            if self.progressBar.value() >= self.progressBar.maximum():
                timer.stop()
            self.progressBar.setValue(self.progressBar.value()+5)
            self.progressBar.setFormat("当前人数:{}/总人数%m".format(self.progressBar.value()-self.progressBar.minimum()))
        timer.timeout.connect(timer_func)  #使用信号
        timer.start(1000)  #每隔1s
        #信号
        self.progressBar.valueChanged.connect(lambda val:print(val))


if __name__ == '__main__':
    app = QApplication(sys.argv)

    window = Window()
    window.show()

    sys.exit(app.exec_())