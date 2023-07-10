from PyQt5.Qt import * #刚开始学习可以这样一下导入
import sys

class MyLabel(QLabel):

    def __init__(self,*args,**kwargs):  # 这更加通用
        super().__init__(*args,**kwargs)
        self.setStyleSheet("font-size:22px;")
        self.move(200,200)


    def setSec(self,sec):
        self.setText(str(sec))

    def startMyTimer(self,ms):
         self.timer_id = self.startTimer(ms)

    def timerEvent(self, *args, **kwargs):
        print("python",)
        #1,获取当前的标签内容
        current_sec = int(self.text())
        current_sec -=1
        self.setText(str(current_sec))
        if current_sec == 0:
            self.killTimer(self.timer_id)


#1,创建app
app  = QApplication(sys.argv)

#2，控件的操作：
#创建控件
window = QWidget()

#设置控件
window.setWindowTitle("QObject 之定时器的学习")
window.resize(500,500)

label = MyLabel(window)
label.setSec(5)

label.startMyTimer(500)

#展示控件
window.show()

#3,进入消息循环
sys.exit(app.exec_())
