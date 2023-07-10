#numpy.asarray 类似 numpy.array
import numpy as np

#将列表转换为 ndarray
x = [1, 2, 3]               #x  任意形式的输入参数，可以是，列表, 列表的元组, 元组, 元组的元组, 元组的列表，多维数组
a = np.asarray(x)
print(a)
print("---------------")

#元组转换为 ndarray:
x =  (1,2,3)
a = np.asarray(x)
print (a)
print("---------------")

#将元组列表转换为 ndarray
x = [(1, 2, 3), (4, 5)]
a = np.asarray(x)
print(a)
print("---------------")

#设置了 dtype 参数：
x =  [1,2,3]
a = np.asarray(x, dtype =  float)
print (a)
print("---------------")


#numpy.frombuffer 用于实现动态数组。
#numpy.frombuffer(buffer, dtype = float, count = -1, offset = 0)
# buffer 	可以是任意对象，会以流的形式读入。
# dtype 	返回数组的数据类型，可选
# count 	读取的数据数量，默认为-1，读取所有数据。
# offset 	读取的起始位置，默认为0。
s =  b'Hello World'
a = np.frombuffer(s, dtype =  'S1')
print (a)
print("---------------")


#numpy.fromiter 方法从可迭代对象中建立 ndarray 对象，返回一维数组
# 使用 range 函数创建列表对象
list = range(5)
it = iter(list)               #iter  迭代器
# 使用迭代器创建 ndarray
x = np.fromiter(it, dtype=float)
print(x)

