from PyQt5.Qt import * #刚开始学习可以这样一下导入
import sys

class Btn(QAbstractButton):
    def paintEvent(self, event):
        # print("绘制按钮")
        painter = QPainter(self) # 它里面的参数是 QPaintDevice “纸”，所有控件都可以当做纸，因为QWidget也继承了QPaintDevice
            #这时就有了画家和 纸

        #给画家支笔
        pen = QPen(QColor(100,10,155),10)  #笔的颜色和宽度
        painter.setPen(pen)


        #画家开始画
        painter.drawText(25,25,self.text())

        painter.drawEllipse(0,0,100,100)

#1,创建app
app  = QApplication(sys.argv)


#2，控件的操作：
#创建控件
window = QWidget()


#设置控件
window.setWindowTitle("QAbstractButton的学习")
window.resize(500,500)

btn = Btn(window)

btn.setText("按钮")
btn.resize(100,100)
btn.pressed.connect(lambda :print("点击了这个按钮"))

#展示控件
window.show()

#3,进入消息循环
sys.exit(app.exec_())