
f = open("yesterday",'r',encoding="utf-8")
# for i in range(5):                                        #选择打印行数
#     print(f.readline())

print(f.readlines())                                     #按行转换成列表

# for line in f.readlines():                                #转换成列表   然后再打印出来           需要把文件导入内存，然后再操作，适用于小文件
#     print(line)


#读文件

#date= open("yesterday",encoding="utf-8").read()     #只是将文件全部读出来
# f = open("yesterday",'r',encoding="utf-8")           #文件句柄   包含了文件的内存对象，名字内存大小，内存位置   给了   f
# data = f.read()
# data2 = f.read()
# print(data)
# print('------------------------------')
# print(data2)                                      #文件读取结束  不能再读了

'''
              #写文件     会新建一个文件
f = open("write_file",'w',encoding="utf-8")           #文件句柄   包含了文件的内存对象，名字内存大小，内存位置   给了   f
f.write("123")
f.write("我爱你\n")
f.write("一全都听你的")
'''

'''
#a = append 追加               不修改原文件  只能再后面追加
f = open("write_file",'a',encoding="utf-8")
f.write("\n二给你最好的\n")
f.write("三你就是最好的")
f.close()

f = open("write_file",'r',encoding="utf-8")
date=f.read()
print(date)
'''
'''
f = open("yesterday",'r',encoding="utf-8")
for i in range(1):                                        #选择打印行数
    print(f.readline())

#print(f.readlines())                                     #按行转换成列表

for line in f.readlines():                                #转换成列表   然后再打印出来           需要把文件导入内存，然后再操作，适用于小文件
    print(line)                                           #因为中间有\n换行所以打印出来间隔一行   这时可以用    print(line.strp())去掉两边的空格和换行
'''
'''
#适用于小文件
f = open("yesterday",'r',encoding="utf-8")
for index,line in enumerate(f.readlines()):                    #引入序号，判断更改    将txt文件写出来
    if index == 9:
        print('----我是分割线----------')
        continue
    print(line.strip())
'''
'''
#改进，好用的方法   高效率     for line in f：    print（line）
f = open("yesterday",'r',encoding="utf-8")
#for line in  f:                                        #迭代器   内存中只保存一行
  #  print(line)
count = 0
for line in f:
    if count == 9:
        print('----我是分割线----------')
        count += 1
        continue
    print(line)
    count +=1
'''
'''
f = open("yesterday",'r',encoding="utf-8")
print(f.tell())                                           #相当于读到位置0    打印光标
print(f.readline())
print(f.readline())
print(f.tell())                                            #相当于读到位置32  也就是打印到32个字符
f.seek(0)                                                #相当于回到第一行，也可以是其他地方  可以修改括号里的值
print(f.readline())

print(f.encoding)                                        #输出编码格式
print(f.fileno())                                       #输出文件编号       不常用
print(f.name)                                            #输出文件名字
print(f.flush())                                        #zhongyao          修改的东西     本来是缓存在内存里的      输入后执行，可以立即将修改的内容刷新到硬盘中  ，

f = open("write_file",'a',encoding="utf-8")
f.truncate(20)                                          #截断，从头往后截20字符  保留前面的
'''
'''
f = open("yesterday",'r+',encoding="utf-8")              #文件句柄 读 写     只能写到最后一行                        读写模式   常用
#f = open("yesterday",'w+',encoding="utf-8")            #文件句柄 写 读     也只能在最后一行写   内存是不允许的        不常用
#f = open("yesterday",'a+',encoding="utf-8")            #文件句柄 追加读写
print(f.read())
f.write("我爱北京")
'''

# f = open("yesterday2",'wb')           #文件句柄  二进制文件      1.网络传输  只能用二进制传输 2.操作二进制文件，如视频
#
# f.write("hello binary\n".encode())         #转成二进制   存进入磁盘，但是我们看见的不是二进制码
# f.close()







