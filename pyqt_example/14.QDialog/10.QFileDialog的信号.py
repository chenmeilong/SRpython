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
        fileDialog.setNameFilters(["ALL(*.*)", "Python文件(*.py)", "Images(*.png *.jpg)"])

        fileDialog.currentChanged.connect(lambda path: print("当前路径改变", path))
        fileDialog.currentUrlChanged.connect(lambda url: print("当前Qurl改变", url))

        fileDialog.directoryEntered.connect(lambda path:print("进入目录",path))
        fileDialog.directoryUrlEntered.connect(lambda url:print("进入目录（QUrl）",url))

        fileDialog.filterSelected.connect(lambda filter:print("过滤器被选中",filter))

        #这里用到前面的文件模式：
        fileDialog.setFileMode(QFileDialog.ExistingFiles)   #多选文件模式

        fileDialog.fileSelected.connect(lambda file: print("单个文件被选中- 字符串", file))
        fileDialog.filesSelected.connect(lambda file: print("多个文件被选中- 列表[字符串]", file))
        fileDialog.urlSelected.connect(lambda file: print("单个文件被选中- url ", file))
        fileDialog.urlsSelected.connect(lambda file: print("多个文件被选中- 列表[url]", file))

        fileDialog.open()

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