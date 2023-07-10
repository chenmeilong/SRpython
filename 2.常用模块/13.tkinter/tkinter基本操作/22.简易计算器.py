from tkinter import *                #简易计算器

master = Tk()

frame = Frame(master)
frame.pack(padx=10, pady=10)

v1 = StringVar()
v2 = StringVar()
v3 = StringVar()

def test(content):
    return content.isdigit()

testCMD = master.register(test)
e1 = Entry(frame, width=10, textvariable=v1, validate="key", \
           validatecommand=(testCMD, '%P')).grid(row=0, column=0)     #'%P'  该 值 为输入框最新内容

Label(frame, text="+").grid(row=0, column=1)

e2 = Entry(frame, width=10, textvariable=v2, validate="key", \
           validatecommand=(testCMD, '%P')).grid(row=0, column=2)

Label(frame, text="=").grid(row=0, column=3)

e3 = Entry(frame, width=10, textvariable=v3, state="readonly").grid(row=0, column=4)    #只读网格线

def calc():
    result = int(v1.get()) + int(v2.get())
    v3.set(str(result))                  #转成字符串

Button(frame, text="计算结果", command=calc).grid(row=1, column=2, pady=5)

mainloop()
