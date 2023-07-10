from tkinter import *        #text组件   显示和处理多行文本   适用于处理多种任务

root = Tk()

text = Text(root, width=30, height=2)
text.pack()

text.insert(INSERT, "I love \n")      #插入光标当前索引
text.insert(END, "FishC.com!")        #

mainloop()
