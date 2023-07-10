from tkinter import *

master = Tk()

LANGS = [
    ("Python", 1),
    ("Perl", 2),
    ("Ruby", 3),
    ("Lua", 4)]

v = IntVar()       #那么变量v被赋值为1，否则为0   # IntVar() 数据转换 取整
v.set(1)
for lang, num in LANGS:
    b = Radiobutton(master, text=lang, variable=v, value=num, indicatoron=False)
    b.pack(fill=X)

mainloop()
