import multiprocessing

from PyQt5.Qt import * #刚开始学习可以这样一下导入
import sys,time

def create_ui_show(rate):
    app  = QApplication(sys.argv)
    progressBar = QProgressBar()

    timer = QTimer()
    def test():
        print(rate)
        progressBar.setValue(rate.value+1)
        if rate.value == 99:
            sys.exit(0)
    timer.timeout.connect(test)
    timer.start(100)

    progressBar.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    rate = multiprocessing.Value("d",0)  #初始的rate.value = 0    multiprocessing.Value可以在不同进程之间共享数据

    process1 = multiprocessing.Process(target=create_ui_show,args=(rate,))
    process1.start()

    for i in range(100):
        time.sleep(0.1)
        rate.value = i
    print("下载完成")