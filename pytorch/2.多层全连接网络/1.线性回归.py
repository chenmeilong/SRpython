import torch
import numpy as np
from torch.autograd import Variable

torch.manual_seed(2017)
# 读入数据 x 和 y
x_train = np.array([[3.3], [4.4], [5.5], [6.71], [6.93], [4.168],
                    [9.779], [6.182], [7.59], [2.167], [7.042],
                    [10.791], [5.313], [7.997], [3.1]], dtype=np.float32)

y_train = np.array([[1.7], [2.76], [2.09], [3.19], [1.694], [1.573],
                    [3.366], [2.596], [2.53], [1.221], [2.827],
                    [3.465], [1.65], [2.904], [1.3]], dtype=np.float32)
# 画出图像
import matplotlib.pyplot as plt
plt.plot(x_train, y_train, 'bo')
plt.show()

# 转换成 Tensor
x_train = torch.from_numpy(x_train)
y_train = torch.from_numpy(y_train)

# 定义参数 w 和 b
w = Variable(torch.randn(1), requires_grad=True)  # 随机初始化  变量
b = Variable(torch.zeros(1), requires_grad=True)  # 使用 0 进行初始化

# 构建线性回归模型
x_train = Variable(x_train)
y_train = Variable(y_train)
def linear_model(x):
    return x * w + b
y_ = linear_model(x_train)

# 经过上面的步骤我们就定义好了模型，在进行参数更新之前，我们可以先看看模型的输出结果长什么样
print(w,b)  #查看w和b初始化的值
plt.plot(x_train.data.numpy(), y_train.data.numpy(), 'bo', label='real')
plt.plot(x_train.data.numpy(), y_.data.numpy(), 'ro', label='estimated')   #预测结果
plt.show()
# 红色的点表示预测值

# 计算误差
def get_loss(y_, y):
    return torch.mean((y_ - y_train) ** 2)
loss = get_loss(y_, y_train)

# 打印一下看看 loss 的大小
print("loss:",loss)

# 定义好了误差函数，接下来我们需要计算 w 和 b 的梯度了，这时得益于 PyTorch 的自动求导，我们不需要手动去算梯度，有兴趣的同学可以手动计算一下，w 和 b 的梯度分别是
# 自动求导
loss.backward()
# 查看 w 和 b 的梯度
print(w.grad)
print(b.grad)

# 更新一次参数
w.data = w.data - 1e-2 * w.grad.data
b.data = b.data - 1e-2 * b.grad.data

# 更新完成参数之后，我们再一次看看模型输出的结果
y_ = linear_model(x_train)
plt.plot(x_train.data.numpy(), y_train.data.numpy(), 'bo', label='real')
plt.plot(x_train.data.numpy(), y_.data.numpy(), 'ro', label='estimated')  #反向更新一次的图像
plt.show()

# 从上面的例子可以看到，更新之后红色的线跑到了蓝色的线下面，没有特别好的拟合蓝色的真实值，所以我们需要在进行几次更新
for e in range(10):  # 进行 10 次更新
    y_ = linear_model(x_train)   #     y_ 表示输出  预测值
    loss = get_loss(y_, y_train)

    w.grad.zero_()  # 记得归零梯度
    b.grad.zero_()  # 记得归零梯度
    loss.backward()  #反向求导 即是 更新 梯度

    w.data = w.data - 1e-2 * w.grad.data  # 更新 w
    b.data = b.data - 1e-2 * b.grad.data  # 更新 b
    print('epoch: {}, loss: {}'.format(e, loss.item()))

y_ = linear_model(x_train)
plt.plot(x_train.data.numpy(), y_train.data.numpy(), 'bo', label='real')
plt.plot(x_train.data.numpy(), y_.data.numpy(), 'ro', label='estimated')
plt.show()

# 经过 10 次更新，我们发现红色的预测结果已经比较好的拟合了蓝色的真实值。

