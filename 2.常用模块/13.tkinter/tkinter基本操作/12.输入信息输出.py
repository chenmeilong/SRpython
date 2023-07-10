from tkinter import *     #tkinter 3种布局 pack（）  gird（）表格形式管理组件位置  row行 column列    place（）

root = Tk()

Label(root, text="作品：").grid(row=0, column=0)
Label(root, text="作者：").grid(row=1, column=0)

e1 = Entry(root)
e2 = Entry(root)
e1.grid(row=0, column=1, padx=10, pady=5)
e2.grid(row=1, column=1, padx=10, pady=5)

def show():
    print("作品：《%s》" % e1.get())
    print("作品：%s" % e2.get())

Button(root, text="获取信息", width=10, command=show)\
             .grid(row=3, column=0, sticky=W, padx=10, pady=5)              #\换行标志   sticky=W位置
Button(root, text="退出", width=10, command=root.quit)\
             .grid(row=3, column=1, sticky=E, padx=10, pady=5)

mainloop()
