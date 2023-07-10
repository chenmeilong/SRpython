from PyQt5.Qt import * #刚开始学习可以这样一下导入
import sys


class Window(QWidget):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.setWindowOpacity(0.5)
        self.setWindowTitle("顶层窗口操作案例")
        self.resize(500,500)
        self.setup_ui()
        self.move_flags = False   # 主要是为了防止，鼠标追踪导致出现问题
        # self.setMouseTracking(True)
    def setup_ui(self):
        self.btn_width = 40
        self.btn_height = 25
        self.top_margin = 0
        self.add_close_max_mini_contrl()
    def add_close_max_mini_contrl(self):
        #添加三个子控件 - 窗口的右上角
        self.close_btn = QPushButton(self)
        self.close_btn.setText("关闭")  #暂时以文本呈现
        self.close_btn.resize(self.btn_width,self.btn_height)

        self.max_btn = QPushButton(self)
        self.max_btn.setText("最大")
        self.max_btn.resize(self.btn_width,self.btn_height)

        self.mini_btn = QPushButton(self)
        self.mini_btn.setText("最小")
        self.mini_btn.resize(self.btn_width,self.btn_height)
    def resizeEvent(self, QResizeEvent):
        print("窗口大小发生变化")
        close_btn_width = self.close_btn.width()
        window_width  = self.width()
        close_btn_x = window_width - close_btn_width
        close_btn_y = self.top_margin
        self.close_btn.move(close_btn_x,close_btn_y)

        max_btn_x = close_btn_x -self.max_btn.width()
        max_btn_y = self.top_margin
        self.max_btn.move(max_btn_x,max_btn_y)

        mini_btn_x = max_btn_x -self.mini_btn.width()
        mini_btn_y = self.top_margin
        self.mini_btn.move(mini_btn_x,mini_btn_y)
    def mousePressEvent(self, event):
        if  event.button() == Qt.LeftButton:
            self.move_flags = True
            self.mouse_x = event.globalX()
            self.mouse_y = event.globalY()

            self.window_x = self.x()
            self.window_y = self.y()
    def mouseMoveEvent(self, event):
        if self.move_flags:
            self.move_x = event.globalX() -self.mouse_x
            self.move_y = event.globalY() -self.mouse_y
            self.move(self.window_x + self.move_x,self.window_y +self.move_y)
    def mouseReleaseEvent(self, event):
        self.move_flags = False

#1,创建app
app  = QApplication(sys.argv)

#2，控件的操作：

window = Window(flags=Qt.FramelessWindowHint)

def close_window_slot():
    window.close()
def max_normal_window_slot():   # 最大化或者正常
    if window.isMaximized():
        window.showNormal()
        window.max_btn.setText("最大")
    else:
        window.showMaximized()
        window.max_btn.setText("恢复")
def mini_window_slot():
    window.showMinimized()

window.close_btn.pressed.connect(close_window_slot)
window.max_btn.pressed.connect(max_normal_window_slot)
window.mini_btn.pressed.connect(mini_window_slot)


#展示控件
window.show()

#3,进入消息循环
sys.exit(app.exec_())