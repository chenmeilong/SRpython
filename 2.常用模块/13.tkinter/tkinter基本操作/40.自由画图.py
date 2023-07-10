from tkinter import *

root = Tk()

w = Canvas(root, width=400, height=200)
w.pack()

def paint(event):
    x1, y1 = (event.x - 1), (event.y - 1)
    x2, y2 = (event.x + 1), (event.y + 1)
    w.create_oval(x1, y1, x2, y2, fill="red")     #画圆

w.bind("<B1-Motion>", paint)   #"<B1-Motion>"鼠标左键拖动  事件

Label(root, text="按住鼠标左键并移动，开始绘制你的理想蓝图吧......").pack(side=BOTTOM)

mainloop()
