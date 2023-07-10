#NumPy 提供了线性代数函数库 linalg，该库包含了线性代数所需的所有功能
# dot 	两个数组的点积，即元素对应相乘。
# vdot 	两个向量的点积
# inner 	两个数组的内积
# matmul 	两个数组的矩阵积
# determinant 	数组的行列式
# solve 	求解线性矩阵方程
# inv 	计算矩阵的乘法逆矩阵
import numpy.matlib
import numpy as np

# numpy.dot() 对于两个一维的数组，计算的是这两个数组对应下标元素的乘积和(数学上称之为内积)；对于二维数组，计算的是两个数组的矩阵乘积；
# 对于多维数组，它的通用计算公式如下，即结果数组中的每个元素都是：数组a的最后一维上的所有元素与数组b的倒数第二位上的所有元素的乘积和：
# dot(a, b)[i,j,k,m] = sum(a[i,j,:] * b[k,:,m])。
a = np.array([[1,2],[3,4]])
b = np.array([[11,12],[13,14]])
print(np.dot(a,b))
#[[1*11+2*13, 1*12+2*14],[3*11+4*13, 3*12+4*14]]
print("---------------")


#numpy.vdot() 函数是两个向量的点积。 如果第一个参数是复数，那么它的共轭复数会用于计算。 如果参数是多维数组，它会被展开。
a = np.array([[1, 2], [3, 4]])
b = np.array([[11, 12], [13, 14]])
# vdot 将数组展开计算内积
print(np.vdot(a, b))
print("---------------")


#numpy.inner() 函数返回一维数组的向量内积。对于更高的维度，它返回最后一个轴上的和的乘积。
print (np.inner(np.array([1,2,3]),np.array([0,1,0])))
# 等价于 1*0+2*1+3*0
print("---------------")

#多维数组实例
a = np.array([[1, 2], [3, 4]])
print('数组 a：')
print(a)
b = np.array([[11, 12], [13, 14]])
print('数组 b：')
print(b)
print('内积：')
print(np.inner(a, b))
# 内积计算式为：
# 1*11+2*12, 1*13+2*14
# 3*11+4*12, 3*13+4*14
print("---------------")


#numpy.matmul 函数返回两个数组的矩阵乘积。 虽然它返回二维数组的正常乘积，但如果任一参数的维数大于2，则将其视为存在于最后两个索引的矩阵的栈，并进行相应广播。
a = [[1,0],[0,1]]
b = [[4,1],[2,2]]
print (np.matmul(a,b))
#二维和一维运算：
a = [[1,0],[0,1]]
b = [1,2]
print (np.matmul(a,b))
print (np.matmul(b,a))
#维度大于二的数组 ：
a = np.arange(8).reshape(2,2,2)
b = np.arange(4).reshape(2,2)
print (np.matmul(a,b))
print("---------------")


#numpy.linalg.det() 函数计算输入矩阵的行列式。
a = np.array([[1, 2], [3, 4]])
print(np.linalg.det(a))

b = np.array([[6,1,1], [4, -2, 5], [2,8,7]])
print (b)
print (np.linalg.det(b))
print (6*(-2*7 - 5*8) - 1*(4*7 - 5*2) + 1*(4*8 - -2*2))
print("---------------")


#numpy.linalg.solve() 函数给出了矩阵形式的线性方程的解。
#numpy.linalg.inv() 函数计算矩阵的乘法逆矩阵。
x = np.array([[1,2],[3,4]])
y = np.linalg.inv(x)
print (x)
print (y)
print (np.dot(x,y))
print("---------------")
#现在创建一个矩阵A的逆矩阵:
a = np.array([[1, 1, 1], [0, 2, 5], [2, 5, -1]])
print('数组 a：')
print(a)
ainv = np.linalg.inv(a)
print('a 的逆：')
print(ainv)
print('矩阵 b：')
b = np.array([[6], [-4], [27]])
print(b)
print('计算：A^(-1)B：')
x = np.linalg.solve(a, b)
print(x)
# 这就是线性方向 x = 5, y = 3, z = -2 的解

