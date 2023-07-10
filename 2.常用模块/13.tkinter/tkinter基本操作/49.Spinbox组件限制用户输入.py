from tkinter import *             # Spinbox  组件      和entry类似  新增   范围或元组来 允许用户输入

root = Tk()

#w = Spinbox(root, from_=0, to=10)         #范围
w = Spinbox(root, values=("小甲鱼", "~风介~", "wei_Y", "戴宇轩"))      #元组
w.pack()

mainloop()
