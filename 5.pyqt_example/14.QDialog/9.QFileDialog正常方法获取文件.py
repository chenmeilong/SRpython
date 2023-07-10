from PyQt5.Qt import *  # 刚开始学习可以这样一下导入
import sys


class Window(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("QFileDialog的学习")
        self.resize(400, 400)
        self.set_ui()

    def set_ui(self):
        fileDialog = QFileDialog(self, "选择一个文件", "./", "ALL(*.*);;Python文件(*.py);;Images(*.png *.jpg)")

        # fileDialog.setAcceptMode(QFileDialog.AcceptSave)  # 此时就是保存的对话框了    加这一行设置为保存
        # fileDialog.setDefaultSuffix("txt")  # 设置默认保存后缀名


        # fileDialog.setFileMode(QFileDialog.Directory)  #设置成打开文件夹  可以设置成打开多个文件



        fileDialog.setNameFilters(["ALL(*.*)", "Python文件(*.py)", "Images(*.png *.jpg)"]) #会把之前的给覆盖了  #过滤器

        # fileDialog.setViewMode(QFileDialog.List)#设置信息的详细程度
        # fileDialog.setViewMode(QFileDialog.Detail)  #设置信息的详细程度

        #设置标签名称
        # fileDialog.setLabelText(QFileDialog.Accept, "Sure")
        # fileDialog.setLabelText(QFileDialog.Reject, "Cancel")






        fileDialog.open()  # 此时如何拿到用户选择的文件呢，这里要借助信号了。

        fileDialog.fileSelected.connect(lambda file: print(file))   #接收用户的 选择

        btn = QPushButton(self)
        btn.setText("按钮")
        btn.move(0, 300)
        btn.clicked.connect(self.btn_clicked_slot)

    def btn_clicked_slot(self):
        pass

    def test(self):
        pass


if __name__ == '__main__':
    app = QApplication(sys.argv)

    window = Window()
    window.show()

    sys.exit(app.exec_())