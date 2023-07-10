import tkinter as tk  # 使用Tkinter前需要先导入
import tkinter.colorchooser  # 要使用colorchooser先要导入模块

window = tk.Tk()

def callback():
    fileName = tkinter.colorchooser.askcolor()    #askcolor(color,title，parent)  #初始化颜色  ，标题栏文本  ，parent=w子窗口开
    print(fileName)     #返回rgb颜色值 十进制颜色值

tk.Button(window, text="选择颜色", command=callback).pack()

window.mainloop()
