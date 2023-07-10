from tkinter import *   #radiobutton组件  单选按钮

master = Tk()

v = IntVar()       #那么变量v被赋值为1，否则为0   # IntVar() 数据转换 取整

Radiobutton(master, text="One", variable=v, value=1).pack(anchor=W)
Radiobutton(master, text="Two", variable=v, value=2).pack(anchor=W)
Radiobutton(master, text="Three", variable=v, value=3).pack(anchor=W)

mainloop()
