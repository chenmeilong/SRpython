from PyQt5.Qt import * #刚开始学习可以这样一下导入
import sys

class Window(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("日期时间 的学习")
        self.resize(400,400)
        self.set_ui()
    def set_ui(self):
        ############################QDataTime的构造方法###############################
        # dt = QDateTime(2018,12,20,12,30)
        # print(dt)
        dt = QDateTime.currentDateTime()  #它是个静态方法  获取当前时间
        print(type(dt))
        print(dt)

        ############################QDataTime的构造方法###############################

        ############################调整时间###############################
        new_dt = dt.addYears(2)
        print(new_dt)
        ############################调整时间###############################

        ############################计算时间差###############################
        #距离时间标准时间的偏差
        print(dt.offsetFromUtc()//3600)  #得到时间差的小时数
        ############################计算时间差###############################

        date = QDate.currentDate()
        print(date)
        print(date.year())

        time = QTime.currentTime()
        print(time.hour())

if __name__ == '__main__':
    app =QApplication(sys.argv)

    window = Window()
    window.show()

    sys.exit(app.exec_())