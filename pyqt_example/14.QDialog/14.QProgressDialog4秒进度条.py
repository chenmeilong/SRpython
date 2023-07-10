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
        #它会自动的弹出  如果在4s 内进度条已经走完了，那么它就不会弹出了

        # #这个时间是可以修改的。
        # progressDialog.setMinimumDuration(0)

        progressDialog.setValue(50)  #4s 后会被显示的进度条

        for i in range(1,101):
            progressDialog.setValue(i) #4s 后不会被显示

        progressDialog.open(lambda: print("对话框被取消"))  #大家对话框




if __name__ == '__main__':
    app =QApplication(sys.argv)

    window = Window()
    window.show()

    sys.exit(app.exec_())