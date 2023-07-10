from PyQt5.Qt import * #刚开始学习可以这样一下导入
import sys


class Btn(QPushButton):
    # def hitButton(self, QPoint):
    def hitButton(self, point):
        print(point)  #用户点击按钮之后，这个函数会将用户点击的点传出来
        cir_x = self.width()/2
        cir_y = self.height()/2

        hit_x = point.x()
        hit_y = point.y()

        if pow(hit_x-cir_x,2)+pow(hit_y-cir_y,2) >pow(self.width()/2,2):
            return False
        return True
############################画内切圆###############################
    def paintEvent(self, event):
        ###########################################################
        super().paintEvent(event)
        ###########################################################

        painter = QPainter(self) # 它里面的参数是 QPaintDevice “纸”，所有控件都可以当做纸，因为QWidget也继承了QPaintDevice
        #这时就有了画家和 纸

        #给画家支笔
        pen = QPen(QColor(100,10,155),4)  #笔的颜色和宽度
        painter.setPen(pen)


        #画家开始画
        painter.drawEllipse(self.rect()) #这就是两个点，四个值
############################画内切圆###############################


#1,创建app
app  = QApplication(sys.argv)

#2，控件的操作：
#创建控件
window = QWidget()

#设置控件
window.setWindowTitle("QAbstactButton 有效区域")
window.resize(500,500)

btn = Btn(window)
btn.setText("按钮")
btn.pressed.connect(lambda :print("按钮被点击"))
btn.resize(200,200)

#展示控件
window.show()

#3,进入消息循环
sys.exit(app.exec_())