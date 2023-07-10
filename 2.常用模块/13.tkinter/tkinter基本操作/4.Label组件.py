from tkinter import *

root = Tk()

photo = PhotoImage(file="bg.gif")
theLabel = Label(root,
                 text="学Python\n到FishC",
                 justify=LEFT,
                 image=photo,
                 compound=CENTER,        #设置图片和文本的混合模式
                 font=("华康少女字体", 20),
                 fg="white"     #字体颜色
                 )
theLabel.pack()

mainloop()
