import torch
from torch.autograd import Variable
# 尝试构建一个函数 $y = x^2 $，然后求 x=2 的导数。
# 参考输出：4

# 答案
x = Variable(torch.FloatTensor([2]), requires_grad=True)
y = x ** 2
y.backward()
print(x.grad)