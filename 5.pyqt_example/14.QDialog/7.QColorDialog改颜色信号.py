from PyQt5.Qt import *  # 刚开始学习可以这样一下导入
import sys


class Window(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("QColorDialog的学习")
        self.resize(400, 400)
        self.set_ui()

    def set_ui(self):
        color = QColor(20, 20, 200)
        self.color = color
        colorDialog = QColorDialog(color, self)
        self.colorDialog = colorDialog

        self.test()

    def test(self):
        btn = QPushButton(self)
        self.btn = btn
        btn.setText("按钮")
        btn.move(0, 300)
        btn.clicked.connect(self.btn_clicked_slot)

    def btn_clicked_slot(self):
        self.colorDialog.setWindowTitle("请选择一个颜色")

        def colorDialog_selectedColor_slot(color):
            # 创建调色板  改变按钮文本的颜色
            palette = QPalette()
            palette.setColor(QPalette.ButtonText, color)
            self.btn.setPalette(palette)

        # 注意colorSelected  和selectedClor 的区别  后者不是信号

        # self.colorDialog.colorSelected.connect(colorDialog_selectedColor_slot)
        self.colorDialog.currentColorChanged.connect(colorDialog_selectedColor_slot)   #实时改变信号

        self.colorDialog.open()


if __name__ == '__main__':
    app = QApplication(sys.argv)

    window = Window()
    window.show()

    sys.exit(app.exec_())