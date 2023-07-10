import sys

#with open("yesterday2", "r", encoding="utf-8") as f:

with open("yesterday2","r",encoding="utf-8") as f,\
      open("yesterday2", "r", encoding="utf-8") as f2:                    #with打开文件后，操作后，可以自动关闭        可以连续打开 多个文件
    for line in f:
        print(line)

