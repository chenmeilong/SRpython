from PyQt5.Qt import * #刚开始学习可以这样一下导入
import sys

class Window(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("交互状态的的案例")
        self.resize(400,400)
        self.set_ui()
        
    def set_ui(self):
        #添加三个子控件
        label = QLabel(self)
        label.setText("标签")
        label.move(50,50)
        label.hide()  #隐藏标签


        lineEdit = QLineEdit(self)
        # lineEdit.setText("文本框")
        lineEdit.move(50,100)

        btn  = QPushButton(self)
        btn.setText("登录")
        btn.move(50,150)
        btn.setEnabled(False)  #设置它不可用

        def textChanged_slot(arg):
            print("文本框内容改变了",arg)
            # if len(arg):
            #     btn.setEnabled(True)
            # else:
            #     btn.setEnabled(False)
            btn.setEnabled(len(arg))

        lineEdit.textChanged.connect(textChanged_slot)

        def check_slot():
            #1,获取文本框的内容
            content = lineEdit.text()
            #2,判断
            label.show()
            if content == "Zcb":
                label.setText("登录成功！")
            else:
                label.setText("登录失败！")

            label.adjustSize()  # 注：它一定要放在设置文本的后面。

        btn.pressed.connect(check_slot)  #用信号clicked 也可以

if __name__ == '__main__':
    app =QApplication(sys.argv)
    window = Window()
    window.show()
    sys.exit(app.exec_())