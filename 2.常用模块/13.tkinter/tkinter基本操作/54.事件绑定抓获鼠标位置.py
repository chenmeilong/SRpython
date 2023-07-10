from tkinter import *    #事件绑定     抓获单击鼠标的位置

root = Tk()

def callback(event):
    print("点击位置：", event.x, event.y)

frame = Frame(root, width=200, height=200)
frame.bind("<Button-1>", callback)
frame.pack()

mainloop()
