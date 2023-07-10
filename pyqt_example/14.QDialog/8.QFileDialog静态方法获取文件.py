from PyQt5.Qt import * #刚开始学习可以这样一下导入
import sys

class Window(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("QFileDialog的学习")
        self.resize(400,400)
        self.set_ui()

    def set_ui(self):                                                # 分隔是以;;  （两个分号）  为过滤器
        ret =  QFileDialog.getOpenFileName(self,"选择一个py文件","./","ALL(*.*);;Images(*.png *.jpg);;Python文件(*.py)","Python文件(*.py)")
        # ret = QFileDialog.getOpenFileNames(self, "选择一个py文件", "./", "ALL(*.*);;Images(*.png *.jpg);;Python文件(*.py)",
        #                                    "Python文件(*.py)")     #获取多个文件

        # ret = QFileDialog.getSaveFileName(self, "选择一个py文件", "./", "ALL(*.*);;Images(*.png *.jpg);;Python文件(*.py)",
        #                                   "Python文件(*.py)")       #保存文件

        # ret = QFileDialog.getExistingDirectory(self, "选文件夹", "./")   #获取文件夹


        print(ret)
if __name__ == '__main__':
    app =QApplication(sys.argv)

    window = Window()
    window.show()

    sys.exit(app.exec_())