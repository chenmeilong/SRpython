a=[1,2,3]           #可以换字符串
b=[4,5,6]
a.append(7)           #列表追加
x=a+b
print(x)
print(len(x))
print(x[0:3])             #左边开始0，右边开始-1

x.insert(1,8)            #中间插入
print(x)

x[0]=10                 #更换
print(x)

#x.remove(6)            #删除
#del x[7]                 #删除
x.pop(7)                 #删除
print(x)

print(x.index(4))         #查找位置

print(x.count(10))        #查找重复个数


#x.clear()                   #清空列表
print(x)

x.reverse()                 #反转
print(x)

x.sort()                   #排序   ascail排序规则
print(x)


x.extend(b)               #合并
print(x)

del b                   #删列表

y1=x.copy()                #浅copy法1
print(x)
print(y1)

y2=x[:]                    #浅copy法2
print(x)
print(y2)

y3=list(x)                 #浅copy法3
print(x)
print(y3)


'''
z=[1,3,[4,5],7,9]            #复制没有复制子列表  上面的 浅copy
y=z.copy()
print(z)
print(y)
z[0]=2
z[2][0]=8
print(z)
print(y)
'''
'''
import copy                             #调用深copy模块
z=[1,3,[4,5],7,9]
y=copy.deepcopy(z)                   #深copy复制子列表
print(z)
print(y)
z[0]=2
z[2][0]=8
print(z)
print(y)
'''

# for i in x:          #列表循环  打印
#     print(i)
# print(x[0:-1:2])        #列表切片


