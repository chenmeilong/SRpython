from tkinter import *       #事件绑定键盘    组件必须要获取焦点

root = Tk()

def callback(event):
    print(event.keysym)
    #print(repr(event.char))

frame = Frame(root, width=200, height=200)    #takefocus=True 也可以获得焦点    需要tab  转换
frame.bind("<Key>", callback)
frame.focus_set()    #获得焦点
frame.pack()

mainloop()
