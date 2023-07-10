from tkinter import *         #canvas组件  用于显示 和编辑图形  绘制直线 圆形

root = Tk()

w = Canvas(root, width=200, height=100)
w.pack()

w.create_line(0, 50, 200, 50, fill="yellow")                 #线    起始横纵坐标  结束横纵坐标
w.create_line(100, 0, 100, 100, fill="red", dash=(4, 4))      #虚线
w.create_rectangle(50, 25, 150, 75, fill="blue")             #矩形

mainloop()
