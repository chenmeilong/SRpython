from PyQt5.Qt import *  # 刚开始学习可以这样一下导入
import sys

class Window(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("QDateTimeEdit 的学习")
        self.resize(400, 400)
        self.set_ui()

    def set_ui(self):
        dateTimeEdit = QDateTimeEdit(QDateTime.currentDateTime(), self)
        dateTimeEdit.resize(150, 30)
        dateTimeEdit.move(100, 100)

        print(dateTimeEdit.sectionCount())

        btn = QPushButton(self)
        btn.setText("按钮")
        btn.move(0, 300)

        def btn_clicked_slot():
            dateTimeEdit.setCalendarPopup(True)  # 设置日历弹出，
            # 如果觉得日历丑的话，可以如下定制
            # dateTimeEdit.calendarWidget()
        btn.clicked.connect(btn_clicked_slot)

        #事件
        dateTimeEdit.dateTimeChanged.connect(lambda val: print("事件",val))
        dateTimeEdit.dateChanged.connect(lambda val: print(val))
        dateTimeEdit.timeChanged.connect(lambda val: print(val))

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = Window()
    window.show()
    sys.exit(app.exec_())