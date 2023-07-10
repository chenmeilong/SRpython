from tkinter import *

master = Tk()

# 创建一个空列表
theLB = Listbox(master, height=11)    #  Listbox 默认为10个项目  修改  height 可以增加
theLB.pack()

# 往列表里添加数据
for item in range(11):
    theLB.insert(END, item)

mainloop()
