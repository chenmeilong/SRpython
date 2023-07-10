import tkinter as tk

class APP:
    def __init__(self, master):
        frame = tk.Frame(master)        #创建一个框架
        frame.pack(side=tk.BOTTOM, padx=10, pady=10)   #  RIGHT 靠右 TOP/LEFT/BOTTOM    设置按钮与边上的距离

        self.hi_there = tk.Button(frame, text="打招呼", bg="black", fg="white", command=self.say_hi)   #创建按钮
        self.hi_there.pack()    #大小自适应

    def say_hi(self):
        print("互联网的广大朋友们大家好，我是小甲鱼！")

root = tk.Tk()                 #创建主窗口
app = APP(root)                #实例化对象app

root.mainloop()
