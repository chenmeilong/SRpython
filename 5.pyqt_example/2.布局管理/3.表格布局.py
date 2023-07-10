import sys
from PyQt5.QtWidgets import (QWidget, QGridLayout,
    QPushButton, QApplication)
from PyQt5.QtWidgets import QAction
from PyQt5.QtWidgets import qApp
from PyQt5.QtGui import QIcon

class Example(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        grid = QGridLayout()
        grid.setSpacing(5)  # 控件上下距离5

        self.setLayout(grid)
        names = ['Cls', 'Bck', '', 'Close',
                 '7', '8', '9', '/',
                '4', '5', '6', '*',
                 '1', '2', '3', '-',
                '0', '.', '=', '+']
        positions = [(i,j) for i in range(5) for j in range(4)]
        for position, name in zip(positions, names):
            print(position,name)
            if name == '':
                continue
            button = QPushButton(name)
            button.clicked.connect(self.buttonClicked)
            grid.addWidget(button, *position)          #注意加*去元组外面的括号    传参名字和位置

        self.move(300, 150)
        self.setWindowTitle('Calculator')
        self.show()

    def buttonClicked(self):
        sender = self.sender()   #指触发事件的控件，之所以是object类型，是因为button按钮也是一个类
        print(sender.text() + '被按那儿了')


app = QApplication(sys.argv)
ex = Example()
sys.exit(app.exec_())
