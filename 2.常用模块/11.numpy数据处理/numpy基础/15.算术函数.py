#NumPy 算术函数包含简单的加减乘除: add()，subtract()，multiply() 和 divide()。     取倒数
import numpy as np
#加减乘除
a = np.arange(9, dtype = np.float_).reshape(3 ,3)
print ('第一个数组：')
print (a)
print ('第二个数组：')
b = np.array([10 ,10 ,10])
print (b)
print ('两个数组相加：')
print (np.add(a ,b))
print ('两个数组相减：')
print (np.subtract(a ,b))
print ('两个数组相乘：')
print (np.multiply(a ,b))
print ('两个数组相除：')
print (np.divide(a ,b))
print("---------------")

#numpy.reciprocal() 函数返回参数逐元素的倒数。如 1/4 倒数为 4/1。
a = np.array([0.25,  1.33,  1,  100])
print ('我们的数组是：')
print (a)
print ('调用 reciprocal 函数：')
print (np.reciprocal(a))
print("---------------")

#numpy.power() 函数将第一个输入数组中的元素作为底数，计算它与第二个输入数组中相应元素的幂。
a = np.array([10,100,1000])
print ('我们的数组是；')
print (a)
print ('调用 power 函数：')
print (np.power(a,2))
print ('第二个数组：')
b = np.array([1,2,3])
print (b)
print ('再次调用 power 函数：')
print (np.power(a,b))
print("---------------")

#numpy.mod() 计算输入数组中相应元素的相除后的余数。 函数 numpy.remainder() 也产生相同的结果。
a = np.array([10,20,30])
b = np.array([3,5,7])
print ('调用 mod() 函数：')
print (np.mod(a,b))
print ('调用 remainder() 函数：')
print (np.remainder(a,b))