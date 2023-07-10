from tkinter import *             #Toplevel 组件   独立顶级窗口组件   通常用于 额外的对话框  和其他弹窗

root = Tk()

def create():
    top = Toplevel()
    #top.attributes("-alpha", 0.5)  # 设置百分之50透明度
    top.title("FishC Demo")     #  类似与frame 组件  ----------标题

    msg = Message(top, text="I love FishC.com")
    msg.pack()

Button(root, text="创建顶级窗口", command=create).pack()    # 按钮

mainloop()
