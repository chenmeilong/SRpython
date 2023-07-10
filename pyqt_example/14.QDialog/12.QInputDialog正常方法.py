from PyQt5.Qt import * #刚开始学习可以这样一下导入
import sys

class Window(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("QInputDialog的学习")
        self.resize(400,400)
        self.set_ui()


    def set_ui(self):
        inputDialog = QInputDialog(self,Qt.FramelessWindowHint)  #第二个参数flags在QWidget中讲过(这里它是没有边框)

        inputDialog.setLabelText("请输入你的姓名")


        #输入模式
        inputDialog.setInputMode(QInputDialog.TextInput)
        inputDialog.setComboBoxItems(["1","2","3"])   #选项设置

        inputDialog.intValueChanged.connect(lambda val:print("整型数据发生改变",val))
        inputDialog.intValueSelected.connect(lambda val:print("整型数据被选中",val))

        inputDialog.doubleValueChanged.connect(lambda val:print("浮点型数据发生改变",val))
        inputDialog.doubleValueSelected.connect(lambda val:print("浮点型数据被选中",val))

        inputDialog.textValueChanged.connect(lambda val:print("字符串型数据发生改变",val))
        inputDialog.textValueSelected.connect(lambda val:print("字符串型数据被选中",val))


        inputDialog.setComboBoxEditable(True)  #设置可编辑


        #输入模式
        # inputDialog.setInputMode(QInputDialog.DoubleInput)
        # inputDialog.setDoubleRange(9.9,19.9)
        # inputDialog.setDoubleStep(2)
        # inputDialog.setDoubleDecimals(3)



        inputDialog.setOkButtonText("好的")
        inputDialog.setCancelButtonText("不了")

        inputDialog.show()
if __name__ == '__main__':
    app =QApplication(sys.argv)

    window = Window()
    window.show()

    sys.exit(app.exec_())