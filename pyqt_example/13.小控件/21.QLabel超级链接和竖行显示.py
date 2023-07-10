from PyQt5.Qt import *  # 刚开始学习可以这样一下导入
import sys


class Window(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("QLabel的学习")
        self.resize(400, 400)
        self.set_ui()

    def set_ui(self):
        label = QLabel("Label", self)
        label.move(100, 100)
        label.resize(200, 50)
        label.setStyleSheet("background-color:cyan;")

        # 设置文本交互标志  #默认是无法选中标签的。
        # 用鼠标和键盘 可以选中  ，可以编辑
        # label.setTextInteractionFlags(Qt.TextSelectableByMouse | Qt.TextSelectableByKeyboard | Qt.TextEditable)
        # label.setSelection(1, 2)  # 从哪里开始  选多少

        #打开外部链接
        label.setText("<a href='http://python123.io'>Python123</a>")
        label.setOpenExternalLinks(True)
        label.linkActivated.connect(lambda val: print("点击了超链接", val))
        # 注：如果setopenexternLinks  开启的话，就不会触发linkActivated这个信号了，因为已经处理了这个信号
        label.linkHovered.connect(lambda val:print("鼠标在超链接上",val))


        #单词换行
        # label.setText("djafs jafssdjlakf jfaljksdl fjd fasdjfal;fadsjl ")
        # label.setWordWrap(True)  #此时就会换行了
        #内容竖着排列
        # label.setText('\n'.join("123456789"))



if __name__ == '__main__':
    app = QApplication(sys.argv)

    window = Window()
    window.show()

    sys.exit(app.exec_())