from PyQt5.Qt import * #刚开始学习可以这样一下导入
import sys
#我们知道QFormLayout 是两列的（Label 和 Feild ）
#那么如果是想要 三列       就要用到这里的网格布局
class Window(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("QFormLayout 的学习")
        self.resize(400,400)
        self.set_ui()


    def set_ui(self):
        name_label= QLabel("姓名：")
        age_label= QLabel("年龄：")
        sex_label = QLabel("性别:")

        name_lineEdit = QLineEdit()
        age_spinBox = QSpinBox()
        male_radioButton = QRadioButton("男")
        female_radioButton = QRadioButton("女")
        h_layout = QHBoxLayout()                 #box布局
        h_layout.addWidget(male_radioButton)
        h_layout.addWidget(female_radioButton)

        submit_btn = QPushButton("提交")

        formLayout = QFormLayout()   #表单布局


        #添加行   法一
        formLayout.addRow(name_label,name_lineEdit)
        formLayout.addRow(age_label,age_spinBox)
        #添加行   法二    且自动添加快捷方式
        # formLayout.addRow("姓名(&n)",name_lineEdit)  #这里会自动的添加快捷方式
        # formLayout.addRow("年龄(&a)",age_spinBox)

        #添加子布局
        formLayout.addRow(sex_label,h_layout)
        formLayout.addRow(submit_btn)

        # 行的信息
        print(formLayout.rowCount())
        print(formLayout.getWidgetPosition(age_label))


        # #修改行   如果单元格位置被占用，则设置不成功
        # formLayout.setWidget(0,QFormLayout.LabelRole,name_label)
        # formLayout.setWidget(0,QFormLayout.FieldRole,name_lineEdit)
        # formLayout.setWidget(1,QFormLayout.LabelRole,sex_label)
        # formLayout.setLayout(1,QFormLayout.FieldRole,h_layout)

        #移除行  删除控件
        # formLayout.removeRow(1)
        #移除行  不删除控件
        # formLayout.takeRow(0)  #它的返回值是  QFormLayout.TakeRowResult 对象
        #移除一个控件
        formLayout.removeWidget(age_label)

        #标签操作
        #此时，可以根据后面的field 找到 label
        mylabel = formLayout.labelForField(name_lineEdit)
        mylabel.setText("标签操作")


        self.setLayout(formLayout)

if __name__ == '__main__':
    app =QApplication(sys.argv)

    window = Window()
    window.show()

    sys.exit(app.exec_())