# numpy.empty 方法用来创建一个指定形状（shape）、数据类型（dtype）且未初始化的数组
# numpy.empty(shape, dtype = float, order = 'C')
# shape 	数组形状
# dtype 	数据类型，可选
# order 	有"C"和"F"两个选项,分别代表，行优先和列优先，在计算机内存中的存储元素的顺序。
#注意 − 数组元素为随机值，因为它们未初始化。
import numpy as np
x = np.empty([3,2], dtype = int)
print (x)
print("---------------")

#numpy.zeros 创建指定大小的数组，数组元素以 0 来填充：
# 默认为浮点数
x = np.zeros(5)
print(x)
# 设置类型为整数
y = np.zeros((5,), dtype=np.int)
print(y)
# 自定义类型
z = np.zeros((2, 2), dtype=[('x', 'i4'), ('y', 'i4')])
print(z)
print("---------------")

#numpy.ones  创建指定形状的数组，数组元素以 1 来填充：
# 默认为浮点数
x = np.ones(5)
print(x)
# 自定义类型
x = np.ones([2, 2], dtype=int)
print(x)



