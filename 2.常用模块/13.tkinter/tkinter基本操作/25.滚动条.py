from tkinter import *        #scrollbar组件 滚动条

root = Tk()

sb = Scrollbar(root)
sb.pack(side=RIGHT, fill=Y)

lb = Listbox(root, yscrollcommand=sb.set)    #yscrollcommand  设置为    Scrollbar.set()  添加滚动条功能

for i in range(1000):
    lb.insert(END, str(i))

lb.pack(side=LEFT, fill=BOTH)

sb.config(command=lb.yview)          #设置  Scrollbar组件  command=lb.yview

mainloop()
