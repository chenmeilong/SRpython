#tags基础   text的标签
from tkinter import *        #indexes索引用法 insert 插入光标位置    current鼠标坐标接近位置  end文本缓冲区最后一个字符的下一个位置
                              #user-defined marks 自定义mark如：insert、current

root = Tk()

text = Text(root, width=30, height=5)
text.pack()

text.insert(INSERT, "I love FishC.com!")

# text.mark_set("here","1.2")         #text中间   插入数据
# text.insert("here","插")
# text.insert("here","入")
# text.delete("1.0",END)             #解除mark封印

# text.mark_set("here","1.2")         #text中间   插入数据
# text.mark_gravity("here",LEFT)       #在左侧插入
# text.insert("here","插")
# text.insert("here","入")


print(text.get("1.2","1.6"))
print(text.get("1.2","1.end"))   #/字符切片输出

text.tag_add("tag1", "1.7", "1.12", "1.14")
text.tag_add("tag2", "1.7", "1.12", "1.14")
text.tag_config("tag1", background="yellow", foreground="red")    #   foreground前景色
text.tag_config("tag2", foreground="blue")
# text.tag_lower("tag2")       #降低tag2的优先级      tag_raise()  提高优先级
mainloop()
