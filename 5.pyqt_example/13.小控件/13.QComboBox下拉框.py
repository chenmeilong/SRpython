from PyQt5.Qt import * #刚开始学习可以这样一下导入
import sys

class Window(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("QComboBox的学习")
        self.resize(400,400)
        self.set_ui()

    def set_ui(self):
        self.comboBox = QComboBox(self)
        self.comboBox.resize(100,30)
        self.comboBox.move(100,100)
        self.test()

        btn = QPushButton(self)
        btn.setText("按钮")
        btn.move(0,300)

        btn.clicked.connect(lambda: print(self.comboBox.currentText()))


    def test(self):
        ############################添加条目项###############################
        self.comboBox.addItem("xx1")
        self.comboBox.addItem("xx2")
        self.comboBox.addItem(QIcon("view.png"),"xx3")
        self.comboBox.addItems(["xx1","xx2","xx3"])
        ############################添加条目项###############################


        self.comboBox.insertItem(1, QIcon("view.png"), "xxx4")  #插入条目项
        # self.comboBox.insertItems()

        self.comboBox.setItemIcon(2, QIcon("view_off.png"))  #设置条目
        self.comboBox.setItemText(2, "fsadjffajs")

        # self.comboBox.removeItem(2)                #删除条目项

        self.comboBox.insertSeparator(2)      #设置分割线


        self.comboBox.setCurrentIndex(3)  #设置当前选项


        self.comboBox.setMaxCount(10)     #显示限制
        self.comboBox.setMaxVisibleItems(5)  #一屏可以显示的个数

        # ########################################################
        self.comboBox.setEditable(True)
        self.comboBox.setCompleter(QCompleter(["xx1","xx2","xx3"]))
        #注意：完成器中的内容一般要和 下拉框中的条目一致  ，这样可以达到快速的匹配


        # ############################常规操作###############################
        # self.comboBox.setEditable(True)
        # self.comboBox.setDuplicatesEnabled(True)  # 设置可重复
        # self.comboBox.setFrame(False)  # 将框架去掉
        # self.comboBox.setIconSize(QSize(60, 60))
        # ############################尺寸调整策略###############################
        # self.comboBox.setSizeAdjustPolicy(QComboBox.AdjustToContents)  # 参照最长的长度
        ############################尺寸调整策略###############################
        # self.comboBox.clear()  #清空所有项目
        # self.comboBox.clearEditText()  #清空编辑的文本
        ############################常规操作###############################


        ############################信号###############################
        #1 self.comboBox.activated.connect(lambda val:print("条目被激活",val))

        # self.comboBox.activated.connect(lambda val:print("条目被激活",self.comboBox.itemText(val)))
        #2 self.comboBox.activated[str].connect(lambda val:print("条目被激活",val))

        # 以上两个信号仅仅是和用户交互的时候发射的信号，如果此时用代码改变，它不会发射信号
        # 如果也想检测到用代码改变的事件，用下面：
        #3
        self.comboBox.currentIndexChanged.connect(lambda val:print("当前索引发生改变",val))
        self.comboBox.currentIndexChanged[str].connect(lambda val:print("当前索引发生改变",val))

        #4
        # self.comboBox.currentTextChanged.connect(lambda val:print("当前文本发生改变",val)) #编辑的时候改变

        #5
        # self.comboBox.editTextChanged.connect(lambda val:print("当前编辑文本发生改变",val))  # 它和 4 差不多

        #6  高亮发生改变     高亮指的是条目的高亮 （选中哪个条目）
        self.comboBox.highlighted.connect(lambda val:print("高亮发生改变",val))



if __name__ == '__main__':
    app =QApplication(sys.argv)

    window = Window()
    window.show()

    sys.exit(app.exec_())