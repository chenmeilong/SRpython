#NumPy 中包含了一个矩阵库 numpy.matlib，该模块中的函数返回的是一个矩阵，而不是 ndarray 对象。
import numpy.matlib
import numpy as np

#numpy.matlib.empty(shape, dtype, order)
# shape: 定义新矩阵形状的整数或整数元组
# Dtype: 可选，数据类型
# order: C（行序优先） 或者
# F（列序优先）
#matlib.empty() 函数返回一个新的矩阵
# 填充为随机数据
print(np.matlib.empty((2, 2)))
print("---------------")

#numpy.matlib.zeros() 函数创建一个以 0 填充的矩阵。
print (np.matlib.zeros((2,2)))
print("---------------")

#numpy.matlib.ones()函数创建一个以 1 填充的矩阵。
print (np.matlib.ones((2,2)))
print("---------------")

#numpy.matlib.eye() 函数返回一个矩阵，对角线元素为 1，其他位置为零。
print (np.matlib.eye(n =  3, M =  4, k =  0, dtype =  float))
print("---------------")

#numpy.matlib.identity() 函数返回给定大小的单位矩阵。
# 大小为 5，类型位浮点型
print (np.matlib.identity(5, dtype =  float))
print("---------------")

#numpy.matlib.rand() 函数创建一个给定大小的矩阵，数据是随机填充的。
print (np.matlib.rand(3,3))
print("---------------")

#矩阵总是二维的，而 ndarray 是一个 n 维数组。 两个对象都是可互换的。
i = np.matrix('1,2;3,4')
print (i)
j = np.asarray(i)
print (j)
k = np.asmatrix (j)
print (k)
