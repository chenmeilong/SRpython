# 利用isWidgetType()判定是否为控件    
from PyQt5.Qt import *  # 刚开始学习可以这样一下导入
import sys


class Window(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("QObject的学习")
        self.resize(400, 400)
        self.set_ui()

    def set_ui(self):
        self.QObject_type_judge()

    def QObject_type_judge(self):
        obj = QObject()
        w = QWidget()

        btn = QPushButton(self) 
        btn.setText("点我")

        label = QLabel()

        obj_list = [obj, w, btn, label]
        for o in obj_list:
            print(o.isWidgetType())    #判断是否为 Widget发子类
            print(o.inherits("QWidget"))            #判断是否为 继承自 Widget

        for widget in self.children():
            if widget.inherits("QPushButton"):
                widget.setStyleSheet("background-color:cyan;")    #查找指定元素 更换底色



if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = Window()
    window.show()
    sys.exit(app.exec_())
'''
    输出：
        False
        True
        True
        True
'''
