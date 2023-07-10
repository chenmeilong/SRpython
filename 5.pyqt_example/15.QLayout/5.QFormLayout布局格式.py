from PyQt5.Qt import *  # 刚开始学习可以这样一下导入
import sys


class Window(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("QFormLayout 的学习")
        self.resize(400, 400)
        self.set_ui()

    def set_ui(self):
        name_label = QLabel("姓名：")
        age_label = QLabel("年龄：")
        sex_label = QLabel("性别:")

        name_lineEdit = QLineEdit()
        age_spinBox = QSpinBox()
        male_radioButton = QRadioButton("男")
        female_radioButton = QRadioButton("女")
        h_layout = QHBoxLayout()
        h_layout.addWidget(male_radioButton)
        h_layout.addWidget(female_radioButton)

        submit_btn = QPushButton("提交")

        formLayout = QFormLayout()

        # 添加行
        formLayout.addRow(name_label, name_lineEdit)
        formLayout.addRow(age_label, age_spinBox)
        # 添加子布局
        formLayout.addRow(sex_label, h_layout)
        formLayout.addRow(submit_btn)

        #行的包装策略
        #默认是左右 摆放，缩放时也不换行
        formLayout.setRowWrapPolicy(QFormLayout.WrapLongRows)   #自动换行
        # formLayout.setRowWrapPolicy(QFormLayout.WrapAllRows)#字段总是位与标签的下方


        #对齐
        #表单的对齐
        formLayout.setFormAlignment(Qt.AlignTop)
        #表单中的标签的对齐
        formLayout.setLabelAlignment(Qt.AlignCenter)

        #间距
        formLayout.setHorizontalSpacing(20)
        formLayout.setVerticalSpacing(60)

        # 字段增长策略
        formLayout.setFieldGrowthPolicy(QFormLayout.FieldsStayAtSizeHint)   #这些领域永远不会超出 其有效大小的提示

        self.setLayout(formLayout)


if __name__ == '__main__':
    app = QApplication(sys.argv)

    window = Window()
    window.show()

    sys.exit(app.exec_())