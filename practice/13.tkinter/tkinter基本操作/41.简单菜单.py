from tkinter import *            #menu组件  菜单

root = Tk()

def callback():
    print("你好~")

menubar = Menu(root)          #创建一个顶级菜单
menubar.add_command(label="hello", command=callback)
menubar.add_command(label="quit", command=root.quit)

root.config(menu=menubar)      #显示菜单

mainloop()
