from PyQt5.Qt import * #刚开始学习可以这样一下导入
import sys

class MyIntValidator(QIntValidator):
    def fixup(self, p_str):
        print("xxx",p_str)
        if len(p_str) == 0 or int(p_str) <18:
            return "18"

class Window(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("验证器QValidator的使用")
        self.resize(400,400)
        self.set_ui()


    def set_ui(self):
        account_lineEdit = QLineEdit(self)
        account_lineEdit.move(100,100)
        #18-100


        ############################QIntValidator 不过它内部完善的不是很充分，我们可以 在它的基础上进行二次开发###############################
        validator = MyIntValidator(18,180)  #限制输入18-180

        account_lineEdit.setValidator(validator)
        ############################QIntValidator 不过它内部完善的不是很充分，我们可以 在它的基础上进行二次开发###############################

        lineEdit = QLineEdit(self)  #主要是让account_lineEdit 完成编辑状态的


if __name__ == '__main__':
    app =QApplication(sys.argv)

    window = Window()
    window.show()

    sys.exit(app.exec_())