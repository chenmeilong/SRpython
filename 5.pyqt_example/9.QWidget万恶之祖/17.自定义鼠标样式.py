from PyQt5.Qt import * #刚开始学习可以这样一下导入
import sys
#1,创建app
app  = QApplication(sys.argv)

#2，控件的操作：
#创建控件
window = QWidget()

#设置控件
window.setWindowTitle("鼠标操作")
window.resize(500,500)

# window.setCursor(Qt.BusyCursor)   #转圈圈的样式

pixmap = QPixmap("icon.png")
new_pixmap = pixmap.scaled(10,10)  #改变图片的尺寸  #注意它是以返回值的形式给出  也可以不改变尺寸，使用原始尺寸
# cursor = QCursor(new_pixmap)    #热点默认在正中心
cursor = QCursor(new_pixmap,0,0)  # 将热点修改为0,0左上角
window.setCursor(cursor)

# window.unsetCursor()  # 重置鼠标形状，使上面的设置的鼠标样式失效

#展示控件
window.show()

#3,进入消息循环
sys.exit(app.exec_())