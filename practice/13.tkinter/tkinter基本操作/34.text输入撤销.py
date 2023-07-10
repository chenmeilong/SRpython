from tkinter import *
import hashlib

root = Tk()

text = Text(root, width=30, height=5, undo=True, autoseparators=False)
text.pack()

text.insert(INSERT, "I love FishC.com!")

def callback(event):
    text.edit_separator()                   #人为插入分隔符

text.bind('<Key>', callback)                #绑定键盘事件

def show():
    text.edit_undo()                  #撤销函数

Button(root, text="撤销", command=show).pack()

mainloop()
