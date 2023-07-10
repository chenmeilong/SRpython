from tkinter import *                #PanedWindow组件 与frame组件类似  新增 允许用户调整应用程序的空间分布

m = PanedWindow(orient=VERTICAL)
m.pack(fill=BOTH, expand=1)

top = Label(m, text="top pane")
m.add(top)

bottom = Label(m, text="bottom pane")
m.add(bottom)

mainloop()
