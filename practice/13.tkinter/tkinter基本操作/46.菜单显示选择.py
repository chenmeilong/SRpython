from tkinter import *      #optionmenu组件

root = Tk()

variable = StringVar()
variable.set("one")

w = OptionMenu(root, variable, "one", "two", "three")
w.pack()

mainloop()
