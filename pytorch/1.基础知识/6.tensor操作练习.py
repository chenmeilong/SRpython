import torch

# **小练习**
# 访问[文档](http://pytorch.org/docs/0.3.0/tensors.html)了解 tensor 更多的 api，实现下面的要求
# 创建一个 float32、4 x 4 的全为1的矩阵，将矩阵正中间 2 x 2 的矩阵，全部修改成2

# 答案
x = torch.ones(4, 4).float()
x[1:3, 1:3] = 2
print(x)
