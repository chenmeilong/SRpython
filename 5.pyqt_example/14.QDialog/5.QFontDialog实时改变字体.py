from PyQt5.Qt import *  # 刚开始学习可以这样一下导入
import sys


class Window(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("QFontDialog的学习")
        self.resize(400, 400)
        self.set_ui()

    def set_ui(self):
        font = QFont()
        font.setFamily("宋体")
        font.setPointSize(14)

        fontDialog = QFontDialog(font, self)
        self.fontDialog = fontDialog

        # fontDialog.setOption(QFontDialog.NoButtons)  #此时就没有下面的按钮了
        fontDialog.setOptions(QFontDialog.NoButtons | QFontDialog.MonospacedFonts)  # 显示等宽字体

        fontDialog.show()

        label = QLabel(self)
        label.move(200,200)
        label.setText("我爱中国")
        def fontDialog_currentFontChanged_slot(font):
            label.adjustSize()
            label.setFont(font)
        fontDialog.currentFontChanged.connect(fontDialog_currentFontChanged_slot)

if __name__ == '__main__':
    app = QApplication(sys.argv)

    window = Window()
    window.show()

    sys.exit(app.exec_())