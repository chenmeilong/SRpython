from tkinter import *       #布局管理  pack（）  按添加顺序排列组件    默认情况下 依次纵向排列

root = Tk()

listbox = Listbox(root)
listbox.pack(fill=BOTH, expand=True)    #fill告诉窗口 给他分配的空间    BOTH表示很响和纵向扩展     expand 告诉组件额外空间也要填满

for i in range(20):
    listbox.insert(END, str(i))

mainloop()
