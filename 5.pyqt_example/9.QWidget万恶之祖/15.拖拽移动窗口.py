from PyQt5.Qt import * #刚开始学习可以这样一下导入
import sys

class Window(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("窗口移动的学习")
        self.resize(400,400)
        self.set_ui()
        self.move_flag = False
    def set_ui(self):
        pass

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:           #鼠标左键按下
            self.move_flag  = True
            # print("鼠标按下")
            QMouseEvent
            #确定两个点，鼠标第一次按下的点，和控件窗口的原始点
            self.mouse_x = event.globalX()
            self.mouse_y = event.globalY()
            self.contrl_window_x = self.x()  # 控件窗口的全局坐标
            self.contrl_window_y = self.y()

    def mouseMoveEvent(self, event):
        if self.move_flag:
            # print("鼠标移动")
            #计算移动向量
            move_x  = event.globalX() - self.mouse_x
            move_y  = event.globalY() - self.mouse_y
            print(move_x,move_y)
            #我们将这个移动向量作用到控件窗口的原始点就行了
            self.move(self.contrl_window_x+move_x,self.contrl_window_y+move_y)

    def mouseReleaseEvent(self, QMouseEvent):
        print("鼠标松开")
        self.move_flag = False


if __name__ == '__main__':
    app =QApplication(sys.argv)

    window = Window()
    window.setMouseTracking(True)
    window.show()

    sys.exit(app.exec_())