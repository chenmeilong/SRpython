from tkinter import *              #text的事件绑定 打开网页
import webbrowser

root = Tk()

text = Text(root, width=30, height=5)
text.pack()

text.insert(INSERT, "I love baidu.com!")

text.tag_add("link", "1.7", "1.16")
text.tag_config("link", foreground="blue", underline=True)

def show_arrow_cursor(event):
    text.config(cursor="arrow")          # 变箭头

def show_xterm_cursor(event):
    text.config(cursor="xterm")          #变光标

def click(event):
    webbrowser.open("http://www.baidu.com")         #打开浏览网页

text.tag_bind("link", "<Enter>", show_arrow_cursor)
text.tag_bind("link", "<Leave>", show_xterm_cursor)
text.tag_bind("link", "<Button-1>", click)

mainloop()
