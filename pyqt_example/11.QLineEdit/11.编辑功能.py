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

############################常用编辑功能###############################
#1
# def btn_clicked_slot():
#     lineEdit.backspace()  #向左删除
#     lineEdit.setFocus()
#2
# def btn_clicked_slot():
#     lineEdit.del_()  #向右删除
#     lineEdit.setFocus()
#3
# def btn_clicked_slot():
#     lineEdit.clear()    #清空  相当于设置空字符串
#     lineEdit.setFocus()
#4  复制
def btn_clicked_slot():
    lineEdit.cursorBackward(True,3)  # 这里要用代码来模拟
    lineEdit.copy()  #复制
    lineEdit.end(False)
    lineEdit.paste()  #粘贴
    lineEdit.setFocus()
#5   剪切
# def btn_clicked_slot():
#     lineEdit.cursorBackward(True,3)  # 这里要用代码来模拟  #True移动选中相关文本
#     lineEdit.cut()  #复制
#     lineEdit.home(False)
#     lineEdit.paste()  #粘贴
#     lineEdit.setFocus()
#6 撤销和重做  undo  redo
#  isUndoAvailable  isRddoAvailable
#7 拖放
# lineEdit2 = QLineEdit(window)
# lineEdit2.setDragEnabled(True)  # 设置可拖放

############################常用编辑功能###############################



btn = QPushButton(window)
btn.setText("按钮")
btn.move(0,300)
btn.clicked.connect(btn_clicked_slot)

#展示控件
window.show()

#3,进入消息循环
sys.exit(app.exec_())