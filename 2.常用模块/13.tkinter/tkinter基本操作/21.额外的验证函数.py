from tkinter import *   #验证函数的 额外 选项   P220

master = Tk()

v = StringVar()

def test(content, reason, name):
    if content == "小甲鱼":
        print("正确！")
        print(content, reason, name)
        return True
    else:
        print("错误！")
        print(content, reason, name)
        return False

testCMD = master.register(test)           #将函数  包起来
e1 = Entry(master, textvariable=v, validate="focusout", \
           validatecommand=(testCMD, '%P', '%v', '%W'))
e2 = Entry(master)
e1.pack(padx=10, pady=10)
e2.pack(padx=10, pady=10)

mainloop()
