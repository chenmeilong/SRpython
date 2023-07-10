from PyQt5.Qt import * #刚开始学习可以这样一下导入
import sys

class Window(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("QCalendarWidget 的学习")
        self.resize(400,400)
        self.set_ui()


    def set_ui(self):
        calendarWidget = QCalendarWidget(self)
        #设置导航条   外观样式调整
        # calendarWidget.setNavigationBarVisible(False)
        # calendarWidget.setFirstDayOfWeek(Qt.Monday)
        # calendarWidget.setGridVisible(True)



        # calendarWidget.setSelectedDate(QDate(-9999,1,1))

        # calendarWidget.setMinimumDate(QDate(1949,10,1))
        # calendarWidget.setMaximumDate(QDate(2049,10,1))
        calendarWidget.setDateRange(QDate(1949,1,1),QDate(2049 ,1,1))  #设置日期范围

        # calendarWidget.setDateEditEnabled(False)  #这样就不能在日期上直接改了
        calendarWidget.setDateEditAcceptDelay(3000)  #3s 编辑结束3s才会跳转过去

        btn = QPushButton(self)
        btn.setText("按钮")
        btn.move(0,300)

        btn.clicked.connect(lambda :print(calendarWidget.yearShown()))
        btn.clicked.connect(lambda :print(calendarWidget.monthShown()))
        btn.clicked.connect(lambda :print(calendarWidget.selectedDate()))

        # 信号
        # calendarWidget.activated.connect(lambda val:print(val))
        # calendarWidget.clicked.connect(lambda val:print(val))
        # calendarWidget.currentPageChanged.connect(lambda y,m:print(y,m))
        calendarWidget.selectionChanged.connect(lambda: print("选中日期改变", calendarWidget.selectedDate()))

    #     btn.clicked.connect(self.btn_clicked_slot)
    # def btn_clicked_slot(self):
    #     # self.calendarWidget.showToday()
    #     # self.calendarWidget.showSelectedDate()
    #     # self.calendarWidget.showNextYear()
    #     # self.calendarWidget.showNextMonth()
    #     # self.calendarWidget.showPreviousMonth()
    #     # self.calendarWidget.showPreviousMonth()
    #     self.calendarWidget.setCurrentPage(2008,8) #跳转
    #     self.calendarWidget.setFocus()



if __name__ == '__main__':
    app =QApplication(sys.argv)

    window = Window()
    window.show()

    sys.exit(app.exec_())