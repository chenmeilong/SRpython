from PyQt5.Qt import * #刚开始学习可以这样一下导入
import sys

class Window(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("QColorDialog的学习")
        self.resize(400,400)
        self.set_ui()


    def set_ui(self):
        color = QColor(20,20,200)
        self.color = color
        colorDialog = QColorDialog(color,self)
        self.colorDialog = colorDialog

        self.test()
    def test(self):
        btn = QPushButton(self)
        btn.setText("按钮")
        btn.move(0,300)
        btn.clicked.connect(self.btn_clicked_slot)

    def btn_clicked_slot(self):
        self.colorDialog.setWindowTitle("选择一个好看的颜色")
        #法一
#########################################
        def colorSelected_slot(color):
            #调色板
            palette = QPalette()
            palette.setColor(QPalette.Background,color)  # 颜色用于 背景
            self.setPalette(palette) #给窗口 设置调好的颜色
            print(color)
        # self.colorDialog.colorSelected.connect(colorSelected_slot)   #点击确定应用
        self.colorDialog.currentColorChanged.connect(colorSelected_slot)        #直接应用
        self.colorDialog.open()
 ############################################

 #法二
 # ###############################################
 #        def colorSelected_slot():
 #            #调色板
 #            palette = QPalette()
 #            palette.setColor(QPalette.Background,self.colorDialog.selectedColor())  # 颜色用于 背景
 #            #给窗口 设置调好的颜色
 #            self.setPalette(palette)
 #        self.colorDialog.open(colorSelected_slot)  #注意它是不会将选择的color 传入槽函数的
 #   #########################################


if __name__ == '__main__':
    app =QApplication(sys.argv)

    window = Window()
    window.show()

    sys.exit(app.exec_())