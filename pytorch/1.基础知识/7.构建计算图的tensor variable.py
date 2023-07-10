import torch
import numpy as np

# ## Variable     我们需要能够构建计算图的 tensor，这就是 Variable。
# tensor 是 PyTorch 中的完美组件，但是构建神经网络还远远不够，我们需要能够构建计算图的 tensor，这就是 Variable。
# Variable 是对 tensor 的封装，操作和 tensor 是一样的，但是每个 Variabel都有三个属性，
# Variable 中的 tensor本身`.data`，对应 tensor 的梯度`.grad`以及这个 Variable 是通过什么方式得到的`.grad_fn`

# 通过下面这种方式导入 Variable
from torch.autograd import Variable

x_tensor = torch.randn(10, 5)
y_tensor = torch.randn(10, 5)

# 将 tensor 变成 Variable
x = Variable(x_tensor, requires_grad=True)  # 默认 Variable 是不需要求梯度的，所以我们用这个方式申明需要对其进行求梯度
y = Variable(y_tensor, requires_grad=True)
z = torch.sum(x + y)

print(z.data)
print(z.grad_fn)   ## 上面我们打出了 z 中的 tensor 数值，同时通过`grad_fn`知道了其是通过 Sum 这种方式得到的

z.backward()    #自动求导
print(x.grad)   # 求 x 和 y 的梯度
print(y.grad)
# 通过`.grad`我们得到了 x 和 y 的梯度，这里我们使用了 PyTorch 提供的自动求导机制，
