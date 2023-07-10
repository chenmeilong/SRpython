from tkinter import *               #message组件    显示多行文本 实现自动换行 调整尺寸

root = Tk()

w1 = Message(root, text="这是一则消息", width=100)
w1.pack()

w2 = Message(root, text="这是一\n则骇人听闻的长长长长长长消息！", width=100)
w2.pack()

mainloop()
