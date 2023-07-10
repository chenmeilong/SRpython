from PyQt5.Qt import * #刚开始学习可以这样一下导入
import sys

class Window(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("QLabel的学习")
        self.resize(800,800)
        self.set_ui()


    def set_ui(self):
        label = QLabel("hello world Life is short ,I learn Python",self)
        label.move(100,100)
        label.resize(200,50)
        label.setStyleSheet("background-color:cyan;")

        #对齐
        # label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)\
        #文本缩进间距
        # label.setIndent(20)
        #文件边距
        label.setMargin(5)


        #文本格式：
        label.setText("<h1>xxx</h1>")
        label.setTextFormat(Qt.PlainText)


        label.setPixmap(QPixmap("view.png"))
        label.adjustSize()  #1,缩放图片 1:1 放置在框内
        label.setScaledContents(True)

        #动图
        movie = QMovie("test.gif")
        label.setMovie(movie)
        label.resize(500,500)
        #设置速度
        movie.setSpeed(300)  #原来的1倍
        movie.start()


        #图形图像画图
        # ###########################################################
        # pic = QPicture()
        # painter = QPainter(pic) # 它里面的参数是 QPaintDevice “纸”，所有控件都可以当做纸，因为QWidget也继承了QPaintDevice
        #     #这时就有了画家和 纸
        #
        # #给画家支笔
        # Qt.BrushStyle
        # brush = QBrush(QColor(100,10,155))  #刷的颜色
        # painter.setBrush(brush)
        #
        # #画家开始画
        # painter.drawEllipse(0,0,200,200)
        #
        # label.setPicture(pic)
        # label.adjustSize()
        # ###########################################################


if __name__ == '__main__':
    app =QApplication(sys.argv)

    window = Window()
    window.show()

    sys.exit(app.exec_())