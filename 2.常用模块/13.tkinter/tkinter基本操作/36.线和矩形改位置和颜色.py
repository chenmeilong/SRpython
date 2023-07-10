from tkinter import *

root = Tk()

w = Canvas(root, width=200, height=100)
w.pack()

line1 = w.create_line(0, 50, 200, 50, fill="yellow")
line2 = w.create_line(100, 0, 100, 100, fill="red", dash=(4, 4))
rect1 = w.create_rectangle(50, 25, 150, 75, fill="blue")

w.coords(line1, 0, 25, 200, 25)              #移动线位置
w.itemconfig(rect1, fill="red")              #修改填充矩形的颜色
w.delete(line2)                              #删除线

Button(root, text="删除全部", command=(lambda x=ALL:w.delete(x))).pack()

mainloop()
