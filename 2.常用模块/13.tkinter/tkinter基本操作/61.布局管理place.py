from tkinter import *        #place 布局管理器   子组件 显示在父组件  正中间     覆盖另一个组件

root = Tk()

photo = PhotoImage(file="logo_big.gif")
Label(root, image=photo).pack()

def callback():
    print("正中靶心！")

Button(root, text="点我", command=callback).place(relx=0.5, rely=0.5, anchor=CENTER)    # rely相对于父组件的位置

mainloop()
