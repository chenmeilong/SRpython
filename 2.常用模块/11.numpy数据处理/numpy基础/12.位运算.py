
import numpy as np
#bitwise_and()函数对数组中整数的二进制形式执行位       与运算。
print('13 和 17 的二进制形式：')
a, b = 13, 17
print(bin(a), bin(b))
print('\n')
print('13 和 17 的位与：')
print(np.bitwise_and(13, 17))
print("---------------")

#bitwise_or()函数对数组中整数的二进制形式执行位        或运算。
a, b = 13, 17
print('13 和 17 的二进制形式：')
print(bin(a), bin(b))
print('13 和 17 的位或：')
print(np.bitwise_or(13, 17))
print("---------------")

#invert() 函数对数组中整数进行位取反运算，即 0 变成 1，1 变成 0。
print('13 的位反转，其中 ndarray 的 dtype 是 uint8：')
print(np.invert(np.array([13], dtype=np.uint8)))
print('\n')
# 比较 13 和 242 的二进制表示，我们发现了位的反转
print('13 的二进制表示：')
print(np.binary_repr(13, width=8))
print('\n')
print('242 的二进制表示：')
print(np.binary_repr(242, width=8))
print("---------------")

#left_shift() 函数将数组元素的二进制形式向左移动到指定位置，右侧附加相等数量的 0。
print('将 10 左移两位：')
print(np.left_shift(10, 2))
print('\n')
print('10 的二进制表示：')
print(np.binary_repr(10, width=8))
print('\n')
print('40 的二进制表示：')
print(np.binary_repr(40, width=8))
#  '00001010' 中的两位移动到了左边，并在右边添加了两个 0。
print("---------------")

#right_shift() 函数将数组元素的二进制形式向右移动到指定位置，左侧附加相等数量的 0。
print('将 40 右移两位：')
print(np.right_shift(40, 2))
print('\n')
print('40 的二进制表示：')
print(np.binary_repr(40, width=8))
print('\n')
print('10 的二进制表示：')
print(np.binary_repr(10, width=8))
#  '00001010' 中的两位移动到了右边，并在左边添加了两个 0。

