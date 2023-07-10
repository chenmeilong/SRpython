import tkinter as tk          #可 重复 建立多个窗口

app = tk.Tk()                                 #创建主窗口
app.title("显示在任务栏的标题")                  #标题  显示在任务栏的

theLabel = tk.Label(app, text="我的第二个窗口程序！")      #label 组件  显示的文本信息  图标或图片
theLabel.pack()                                    #pack自动调节组件自身 尺寸 大小


app.mainloop()      #进入主事件循环
