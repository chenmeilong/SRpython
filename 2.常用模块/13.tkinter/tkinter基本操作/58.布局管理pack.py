from tkinter import *   #挨个排列要使用side选项

root = Tk()

Label(root, text="red", bg="red", fg="white").pack(side=LEFT)
Label(root, text="green", bg="green", fg="black").pack(side=LEFT)
Label(root, text="blue", bg="blue", fg="white").pack(side=LEFT)

mainloop()
