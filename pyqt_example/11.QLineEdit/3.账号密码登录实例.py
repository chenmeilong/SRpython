from PyQt5.Qt import * #刚开始学习可以这样一下导入
import sys,time

class AccountTool:
    SUCCESS = 0
    ACCOUNT_ERROR = 1
    PWD_ERROR = 2
    #真实的开发中，应该是将账号和密码发送给服务器，等待服务器返回

    def check_login(account,pwd):  # 它是个自由方法
        if account != "zcb":
            return AccountTool.ACCOUNT_ERROR

        if pwd != "zcb123":
            return AccountTool.PWD_ERROR
        return AccountTool.SUCCESS


class Window(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("QLineEdit 输出模式的案例")
        self.resize(400,400)
        self.set_ui()

    def set_ui(self):
        self.account_lineEdit = QLineEdit(self)
        self.pwd_lineEdit = QLineEdit(self)
        self.pwd_lineEdit.setEchoMode(QLineEdit.Password )
        self.btn = QPushButton(self)
        self.btn.clicked.connect(self.btn_slot)

        ############################占位字符串的使用###############################
        self.account_lineEdit.setPlaceholderText("请输入账户")
        self.pwd_lineEdit.setPlaceholderText("请输入密码")
        ############################占位字符串的使用###############################


        ############################清空按钮的显示###############################
        self.pwd_lineEdit.setClearButtonEnabled(True)
        ############################清空按钮的显示###############################

        ############################添加操作行为###############################
        action = QAction(self.pwd_lineEdit)

        action.setIcon(QIcon("view_off.png"))
        # self.pwd_lineEdit.addAction(action,QLineEdit.LeadingPosition)
        self.pwd_lineEdit.addAction(action, QLineEdit.TrailingPosition)

        def action_triggered_slot():
            if self.pwd_lineEdit.echoMode() == QLineEdit.Normal:
                print("变为密文")
                self.pwd_lineEdit.setEchoMode(QLineEdit.Password)
                action.setIcon(QIcon("view_off.png"))
            else:
                print("变为明文")
                self.pwd_lineEdit.setEchoMode(QLineEdit.Normal)
                action.setIcon(QIcon("view.png"))

        action.triggered.connect(action_triggered_slot)

    ############################添加操作行为###############################

        ############################自动补全设置###############################

        completer = QCompleter(["zcb","tom","jack","zach"],self.account_lineEdit)
        self.account_lineEdit.setCompleter(completer)

        ############################自动补全设置###############################

    def btn_slot(self):
        account = self.account_lineEdit.text()
        pwd = self.pwd_lineEdit.text()

        state = AccountTool.check_login(account,pwd)
        if state == AccountTool.ACCOUNT_ERROR:
            print("账号错误 ")
            self.account_lineEdit.setText("")  #清空
            self.pwd_lineEdit.setText("") #清空
            self.account_lineEdit.setFocus()  # 让输入框重新获得焦点
            return None   # 让它立马返回
        if state == AccountTool.PWD_ERROR:
            print("密码错误")
            self.pwd_lineEdit.setText("")  #清空密码框
            self.pwd_lineEdit.setFocus()  # 让输入框重新获得焦点
            return None    # 让它立马返回
        if state == AccountTool.SUCCESS:
            print("登录成功")


    def resizeEvent(self, event):
        widget_w = 150
        widget_h = 30
        widget_h_margin = 10
        top_margin = 50

        self.account_lineEdit.resize(widget_w,widget_h)

        self.pwd_lineEdit.resize(widget_w,widget_h)

        self.btn.setText("登   录")
        self.btn.resize(widget_w,widget_h)

        x = (self.width() - widget_w)/2
        y1 = top_margin
        y2 = y1 + widget_h +widget_h_margin
        y3 = y2 + widget_h +widget_h_margin

        self.account_lineEdit.move(x,y1)
        self.pwd_lineEdit.move(x,y2)
        self.btn.move(x,y3)

if __name__ == '__main__':
    app =QApplication(sys.argv)

    window = Window()
    window.show()

    sys.exit(app.exec_())