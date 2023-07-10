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
        # dateTimeEdit = QDateTimeEdit(QDate.currentDate(),self)
        # dateTimeEdit = QDateTimeEdit(QTime.currentTime(),self)
        dateTimeEdit.resize(200,30)
        dateTimeEdit.move(100, 100)

        dateTimeEdit.setDisplayFormat("yy-MM-dd $ hh:mm:ss:zzz")

        print(dateTimeEdit.sectionCount())  #获取section个数

        btn = QPushButton(self)
        btn.setText("按钮")
        btn.move(0, 300)


        #最大和最小日期时间
        dateTimeEdit.setMaximumDateTime(QDateTime(2022, 8, 15, 12, 30))
        dateTimeEdit.setMinimumDateTime(QDateTime().currentDateTime())


        def btn_clicked_slot():
            # btn.clicked.connect(lambda :print(dateTimeEdit.currentSectionIndex())) #获取Section索引
            # btn.clicked.connect(lambda :print(dateTimeEdit.setCurrentSectionIndex(3)))
            # btn.clicked.connect(lambda :print(dateTimeEdit.setCurrentSection(QDateTimeEdit.HourSection)))
            print(dateTimeEdit.sectionText(QDateTimeEdit.HourSection))   #获取指定文本内容

            #获取日期和时间
            print(dateTimeEdit.dateTime())
            print(dateTimeEdit.date())
            print(dateTimeEdit.time())

        btn.clicked.connect(btn_clicked_slot)


if __name__ == '__main__':
    app = QApplication(sys.argv)

    window = Window()
    window.show()

    sys.exit(app.exec_())