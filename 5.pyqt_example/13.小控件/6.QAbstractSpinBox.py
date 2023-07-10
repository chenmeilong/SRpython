from PyQt5.Qt import * #刚开始学习可以这样一下导入
import sys

class MyAbstractSpinBox(QAbstractSpinBox):
    def __init__(self,parent=None,num = '0',*args,**kwargs):  #此处是定义
        super().__init__(parent,*args,**kwargs)  #此处是调用  ，注意区别
        self.lineEdit().setText(num)

    def stepEnabled(self):    #实现上下能调整的方法
        current_num = int(self.text())
        if current_num == 10:
            return QAbstractSpinBox.StepUpEnabled

        elif current_num == 90:
            return QAbstractSpinBox.StepDownEnabled

        elif current_num <10 or current_num >90:
            return QAbstractSpinBox.StepNone

        else:
            return QAbstractSpinBox.StepUpEnabled | QAbstractSpinBox.StepDownEnabled

    def stepBy(self, p_int):  #实现步长调整方法  # 如果上面的方法返回的是有效的话，就会调用这个函数
        print(p_int)
        current_num = int(self.text())+p_int
        ############################关键是如何设置回去###############################
        self.lineEdit().setText(str(current_num))  # 它是个组合控件，左面是个单行输入框，
        ############################关键是如何设置回去###############################

        self.setAccelerated(True)       #设置加速
        print(self.isAccelerated())
        print(self.lineEdit().text())   # 获取文本  获取的是真实的内容



class Window(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("QAbstractSpinBox 的学习")
        self.resize(400,400)
        self.set_ui()
    def set_ui(self):
        abstractSpinBox = MyAbstractSpinBox(self,'50')
        abstractSpinBox.resize(100,30)
        abstractSpinBox.move(100,100)

        # print("只读：",abstractSpinBox.isReadOnly())   #查看只读
        # abstractSpinBox.setReadOnly(True)    #设置只读

        ###########################框内数字对齐###############################
        abstractSpinBox.setAlignment(Qt.AlignCenter)

        print(abstractSpinBox.hasFrame())  # 默认就是True 的  设置周边框架
        # abstractSpinBox.clear()   #清空数字
        # abstractSpinBox.setButtonSymbols(QAbstractSpinBox.NoButtons)  #隐藏右边的按钮

        abstractSpinBox.editingFinished.connect(lambda: print("结束编辑"))


if __name__ == '__main__':
    app =QApplication(sys.argv)
    window = Window()
    window.show()
    sys.exit(app.exec_())