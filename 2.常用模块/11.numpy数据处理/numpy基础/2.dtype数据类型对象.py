# numpy.dtype(object, align, copy)
# object - 要转换为的数据类型对象
# align - 如果为 true，填充字段使其类似 C 的结构体。
# copy - 复制 dtype 对象 ，如果为 false，则是对内置数据类型对象的引用

# 字符 	对应类型
# b	布尔型
# i	(有符号) 整型
# u 	无符号整型 integer
# f	浮点型
# c 	复数浮点型
# m 	timedelta（时间间隔）
# M 	datetime（日期时间）
# O 	(Python) 对象
# S, a 	(byte-)字符串
# U	Unicode
# V 	原始数据 (void)

import numpy as np
# 使用标量类型
dt = np.dtype(np.int32)
print(dt)
print("---------------")

# int8, int16, int32, int64 四种数据类型可以使用字符串 'i1', 'i2','i4','i8' 代替
dt = np.dtype('i4')
print(dt)
print("---------------")

# 字节顺序标注
dt = np.dtype('<i4')    #"<"意味着小端法(最小值存储在最小的地址，即低位组放在最前面)。">"意味着大端法(最重要的字节存储在最小的地址，即高位组放在最前面)
print(dt)
print("---------------")

# 首先创建结构化数据类型
dt = np.dtype([('age',np.int8)])
print(dt)
print("---------------")

# 将数据类型应用于 ndarray 对象
dt = np.dtype([('age',np.int8)])
a = np.array([(10,),(20,),(30,)], dtype = dt)
print(a)
print("---------------")

# 类型字段名可以用于存取实际的 age 列
dt = np.dtype([('age',np.int8)])
a = np.array([(10,),(20,),(30,)], dtype = dt)
print(a['age'])
print("---------------")

#定义一个结构化数据类型 student，包含字符串字段 name，整数字段 age，及浮点字段 marks，并将这个 dtype 应用到 ndarray 对象
student = np.dtype([('name','S20'), ('age', 'i1'), ('marks', 'f4')])    #S表示字符串   f浮点型
print(student)
print("---------------")


student = np.dtype([('name','S20'), ('age', 'i1'), ('marks', 'f4')])
a = np.array([('abc', 21, 50),('xyz', 18, 75)], dtype = student)
print(a)




