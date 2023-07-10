from PyQt5.Qt import * #刚开始学习可以这样一下导入
import sys

class Window(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("QProgressDialog的学习")
        self.resize(400,400)
        self.set_ui()


    def set_ui(self):
        progressDialog = QProgressDialog(self)

        progressDialog.setWindowTitle("HaHa")
        progressDialog.setLabelText("下载进度")
        progressDialog.setCancelButtonText("取消")

        progressDialog.setRange(0,500)

        progressDialog.setValue(490)
        progressDialog.open()

        def timeout_slot():
            print(progressDialog.value())
            if progressDialog.value()+1>= progressDialog.maximum() or progressDialog.wasCanceled():
                timer.stop()
            progressDialog.setValue(progressDialog.value()+1)
            #自动关闭的三个条件：达到最大值，二，自动重置为true  三，可以自动关闭

        timer = QTimer(progressDialog)
        timer.timeout.connect(timeout_slot)
        timer.start(1000)

        progressDialog.canceled.connect(lambda: print("被取消"))


if __name__ == '__main__':
    app =QApplication(sys.argv)

    window = Window()
    window.show()

    sys.exit(app.exec_())