
# ## 把 PyTorch 当做 NumPy 用
# PyTorch 的官方介绍是一个拥有强力GPU加速的张量和动态构建网络的库，其主要构件是张量，所以我们可以把 PyTorch
# 当做 NumPy 来用，PyTorch 的很多操作好 NumPy 都是类似的，但是因为其能够在 GPU 上运行，所以有着比 NumPy 快很多倍的速度。

import torch
import numpy as np

# 创建一个 numpy ndarray
numpy_tensor = np.random.randn(10, 20)
# numpy的ndarray转换到tensor上
pytorch_tensor1 = torch.Tensor(numpy_tensor)


# PyTorch Tensor 使用 GPU 加速
# 我们可以使用以下两种方式将 Tensor 放到 GPU 上
# 第一种方式是定义 cuda 数据类型
dtype = torch.cuda.FloatTensor # 定义默认 GPU 的 数据类型
gpu_tensor = torch.randn(10, 20).type(dtype)
# 第二种方式更简单，推荐使用
gpu_tensor = torch.randn(10, 20).cuda(0) # 将 tensor 放到第一个 GPU 上
# gpu_tensor = torch.randn(10, 20).cuda(1) # 将 tensor 放到第二个 GPU 上
# 使用第一种方式将 tensor 放到 GPU 上的时候会将数据类型转换成定义的类型，而是用第二种方式能够直接将 tensor 放到 GPU 上，类型跟之前保持一致
# 推荐在定义 tensor 的时候就明确数据类型，然后直接使用第二种方法将 tensor 放到 GPU 上


# 而将 tensor 放回 CPU 的操作非常简单
cpu_tensor = gpu_tensor.cpu()
# 我们也能够访问到 Tensor 的一些属性
# 可以通过下面两种方式得到 tensor 的大小
print(pytorch_tensor1.shape)
print(pytorch_tensor1.size())
print(pytorch_tensor1.type())   # 得到 tensor 的数据类型
print(pytorch_tensor1.dim())    # 得到 tensor 的维度
print(pytorch_tensor1.numel())   # 得到 tensor 的所有元素个数

# ## Tensor的操作
# Tensor 操作中的 api 和 NumPy 非常相似，如果你熟悉 NumPy 中的操作，那么 tensor 基本是一致的，下面我们来列举其中的一些操作
x = torch.ones(2, 2)
print(x) # 这是一个float tensor
print(x.type())

# 将其转化为整形
x = x.long()
# x = x.type(torch.LongTensor)
print(x)

# 再将其转回 float
x = x.float()
# x = x.type(torch.FloatTensor)
print(x)

x = torch.randn(4, 3)
print(x)

# 沿着行取最大值
max_value, max_idx = torch.max(x, dim=1)
# 每一行的最大值
print (max_value)
# 每一行最大值的下标
print (max_idx)

# 沿着行对 x 求和
sum_x = torch.sum(x, dim=1)
print(sum_x)

# 增加维度或者减少维度
print(x.shape)
x = x.unsqueeze(0) # 在第一维增加
print(x.shape)

x = x.unsqueeze(1) # 在第二维增加
print(x.shape)

x = x.squeeze(0) # 减少第一维
print(x.shape)
x = x.squeeze() # 将 tensor 中所有的一维全部都去掉
print(x.shape)


x = torch.randn(3, 4, 5)
print(x.shape)
# 使用permute和transpose进行维度交换
x = x.permute(1, 0, 2) # permute 可以重新排列 tensor 的维度
print(x.shape)

x = x.transpose(0, 2)  # transpose 交换 tensor 中的两个维度
print(x.shape)

# 使用 view 对 tensor 进行 reshape
x = torch.randn(3, 4, 5)
print(x.shape)

x = x.view(-1, 5) # -1 表示任意的大小，5 表示第二维变成 5
print(x.shape)

x = x.view(3, 20) # 重新 reshape 成 (3, 20) 的大小
print(x.shape)

x = torch.randn(3, 4)
y = torch.randn(3, 4)

# 两个 tensor 求和
z = x + y
# z = torch.add(x, y)


# 另外，pytorch中大多数的操作都支持 inplace 操作，也就是可以直接对 tensor 进行操作而不需要另外开辟内存空间，方式非常简单，一般都是在操作的符号后面加`_`，比如

x = torch.ones(3, 3)
print(x.shape)
# unsqueeze 进行 inplace
x.unsqueeze_(0)
print(x.shape)

# transpose 进行 inplace
x.transpose_(1, 0)
print(x.shape)

x = torch.ones(3, 3)
y = torch.ones(3, 3)
print(x)

# add 进行 inplace
x.add_(y)
print(x)
