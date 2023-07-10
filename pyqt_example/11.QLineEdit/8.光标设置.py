from PyQt5.Qt import * #刚开始学习可以这样一下导入
import sys
#1,创建app
app  = QApplication(sys.argv)


#2，控件的操作：
#创建控件
window = QWidget()


#设置控件
window.setWindowTitle("光标")
window.resize(500,500)

lineEdit = QLineEdit(window)
lineEdit.move(100,100)
def btn_clicked_slot():
    # lineEdit.cursorBackward(False,2)  # 如果是true 就代表移动时选中相关文本
    # lineEdit.cursorForward(False, 2)  # 如果是true 就代表移动时选中相关文本   右移动光标
    # lineEdit.cursorWordBackward(False)  # True 是选中   移到空格最前面
    # lineEdit.cursorWordForward(False)  #True 是选中     移到空格后面
    # lineEdit.home(False)  # 快速回到行首
    lineEdit.end(False)  # 快速回到行尾
    # lineEdit.setCursorPosition(len(lineEdit.text()) / 2)  # 回到中间位置   指定位置

    print(lineEdit.cursorPosition())  # 获取光标位置
    print(lineEdit.cursorPositionAt(QPoint(105,105)))  #QPoint(105,105)获取屏幕指定位置的光标位置

    lineEdit.setFocus()

btn = QPushButton(window)
btn.setText("按钮")
btn.move(0,300)
btn.clicked.connect(btn_clicked_slot)


#展示控件
window.show()

#3,进入消息循环
sys.exit(app.exec_())