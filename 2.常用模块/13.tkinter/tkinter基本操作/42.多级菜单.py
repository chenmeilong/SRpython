from tkinter import *

root = Tk()

def callback():
    print("你好~")

menubar = Menu(root)          #创建一个顶级菜单

filemenu = Menu(menubar, tearoff=False)       #创建一个下拉菜单 添加到顶级菜单中    tearoff  是否打开菜单
filemenu.add_command(label="打开", command=callback)
filemenu.add_command(label="保存", command=callback)
filemenu.add_separator()                                   #增加分割线
filemenu.add_command(label="退出", command=root.quit)
menubar.add_cascade(label="文件", menu=filemenu)

editmenu = Menu(menubar, tearoff=False)      #创建另一个下拉菜单 添加到顶级菜单中
editmenu.add_command(label="剪切", command=callback)
editmenu.add_command(label="拷贝", command=callback)
editmenu.add_command(label="黏贴", command=callback)
menubar.add_cascade(label="编辑", menu=editmenu)

root.config(menu=menubar)    #显示菜单

mainloop()
