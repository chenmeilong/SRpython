import tkinter as tk

window = tk.Tk()  # 实例化一个窗口
window.title('my window')  # 定义窗口标题
window.geometry('400x300')  # 定义窗口大小

var1 = tk.IntVar()
var2 = tk.IntVar()

def print_selection():
    if ((var1.get()) == 1) & ((var2.get()) == 0):
        print('I love only Python')
    elif ((var1.get()) == 0) & ((var2.get()) == 1):
        print('I love only c++')
    elif ((var1.get()) == 0) & ((var2.get()) == 0):
        print('I do not love eitheer')
    else:
        print('I love both')

c1 = tk.Checkbutton(window, text="python", variable=var1, command=print_selection)
c1.pack()  # variable属性它是与这个控件本身绑定，有自己的值：On和Off值，缺省状态On为1，勾选状态Off为0，
c2 = tk.Checkbutton(window, text="c++", variable=var2, command=print_selection)
c2.pack()
window.mainloop()