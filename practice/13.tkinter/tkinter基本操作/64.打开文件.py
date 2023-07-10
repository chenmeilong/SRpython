import tkinter as tk  # 使用Tkinter前需要先导入
import tkinter.filedialog  # 要使用filedialog 先要导入模块


window = tk.Tk()

def callback():
    fileName = tkinter.filedialog.askopenfilename(filetypes=[("PNG", ".png"), ("GIF", ".gif"), ("JPG", ".jpg"), ("Python", ".py")])
    print(fileName)

tk.Button(window , text="打开文件", command=callback).pack()

window .mainloop()
