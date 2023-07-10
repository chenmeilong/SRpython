from tkinter import *      #鼠标在组件上运行 的  轨迹

root = Tk()

def callback(event):
    print("当前位置：", event.x, event.y)

frame = Frame(root, width=200, height=200)
frame.bind("<Motion>", callback)   #  "<Motion>"事件
frame.pack()

mainloop()
