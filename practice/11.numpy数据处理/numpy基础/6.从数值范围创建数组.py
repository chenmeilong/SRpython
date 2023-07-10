# 循环  等差数列 等比数列   创建数组

#numpy 包中的使用 arange 函数创建数值范围并返回 ndarray 对象
# start 	起始值，默认为0
# stop 	终止值（不包含）
# step 	步长，默认为1
# dtype 	返回ndarray的数据类型，如果没有提供，则会使用输入数据的类型。
import numpy as np
#生成 0 到 5 的数组:
x = np.arange(5)
print(x)
print("---------------")

#设置返回类型位 float:
# 设置了 dtype
x = np.arange(5, dtype =  float)
print (x)
print("---------------")

#设置了起始值、终止值及步长：
x = np.arange(10,20,2)
print (x)
print("---------------")

#numpy.linspace 函数用于创建一个一维数组，数组是一个等差数列构成的
#np.linspace(start, stop, num=50, endpoint=True, retstep=False, dtype=None)
# start 	序列的起始值
# stop 	序列的终止值，如果endpoint为true，该值包含于数列中
# num 	要生成的等步长的样本数量，默认为50
# endpoint 	该值为 ture 时，数列中中包含stop值，反之不包含，默认是True。
# retstep 	如果为 True 时，生成的数组中会显示间距，反之不显示。
# dtype 	ndarray 的数据类型

#用到三个参数，设置起始点为 1 ，终止点为 10，数列个数为 10。
a = np.linspace(1,10,10)
print(a)
print("---------------")

#设置元素全部是1的等差数列
a = np.linspace(1,1,10)
print(a)
print("---------------")

#将 endpoint 设为 false，不包含终止值：
a = np.linspace(10, 20,  5, endpoint =  False)
print(a)
print("---------------")

#设置间距
a = np.linspace(1, 10, 10, retstep=True)
print(a)
# 拓展例子
b = np.linspace(1, 10, 10).reshape([10, 1])
print(b)
print("---------------"*5)


#numpy.logspace 函数用于创建一个于等比数列
#np.logspace(start, stop, num=50, endpoint=True, base=10.0, dtype=None)
# start 	序列的起始值为：base ** start
# stop 	序列的终止值为：base ** stop。如果endpoint为true，该值包含于数列中
# num 	要生成的等步长的样本数量，默认为50
# endpoint 	该值为 ture 时，数列中中包含stop值，反之不包含，默认是True。
# base 	对数 log 的底数。
# dtype 	ndarray 的数据类型


# 默认底数是 10
a = np.logspace(1.0,  2.0, num =  10)    #10^1 开始     10^2结束   个数为10
print (a)
print("---------------")

#将对数的底数设置为 2 :
a = a = np.logspace(0,9,10,base=2)
print (a)




