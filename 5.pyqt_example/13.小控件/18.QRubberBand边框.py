from PyQt5.Qt import *  # 刚开始学习可以这样一下导入
import sys


class Window(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("QRubberBand的案例")
        self.resize(400, 400)
        self.set_ui()

    def set_ui(self):
        for i in range(30):
            checkBox = QCheckBox(self)
            checkBox.setText(str(i))
            checkBox.move(i % 4 * 50, i // 4 * 50)

        self.rubberBand = QRubberBand(QRubberBand.Rectangle, self)


    def mousePressEvent(self, event):
        self.origin_pos = event.pos()
        self.rubberBand.setGeometry(QRect(self.origin_pos, QSize()))  # QSize() 此时为-1 -1
        self.rubberBand.show()


    def mouseMoveEvent(self, event):
        # self.rubberBand.setGeometry(QRect(self.origin_pos,event.pos())) #这里是不可以反着拖的
        self.rubberBand.setGeometry(QRect(self.origin_pos, event.pos()).normalized())  # 这里可以


    def mouseReleaseEvent(self, event):
        rect = self.rubberBand.geometry()
        for child in self.children():
            if rect.contains(child.geometry()) and child.inherits("QCheckBox"):
                # print(child)
                child.toggle()

        self.rubberBand.hide()


if __name__ == '__main__':
    app = QApplication(sys.argv)

    window = Window()
    window.show()

    sys.exit(app.exec_())