# ## 多分类问题
# 前面我们讲过二分类问题，现在处理的问题更加复杂，是一个 10 分类问题，统称为多分类问题，对于多分类问题而言，我们的 loss 函数使用一个更加复杂的函数，叫交叉熵。
#
# ### softmax
# 提到交叉熵，我们先讲一下 softmax 函数，前面我们见过了 sigmoid 函数，如下
# 可以将任何一个值转换到 0 ~ 1 之间，当然对于一个二分类问题，这样就足够了，因为对于二分类问题，如果不属于第一类，那么必定属于第二类，
# 所以只需要用一个值来表示其属于其中一类概率，但是对于多分类问题，这样并不行，需要知道其属于每一类的概率，这个时候就需要 softmax 函数了。

import numpy as np
import torch
from torchvision.datasets import mnist  # 导入 pytorch 内置的 mnist 数据

from torch import nn
from torch.autograd import Variable

# 使用内置函数下载 mnist 数据集
train_set = mnist.MNIST('./data', train=True, download=True)
test_set = mnist.MNIST('./data', train=False, download=True)

# 我们可以看看其中的一个数据是什么样子的
a_data, a_label = train_set[0]
a_data
a_label
# 这里的读入的数据是 PIL 库中的格式，我们可以非常方便地将其转换为 numpy array
a_data = np.array(a_data, dtype='float32')
print(a_data.shape)
# 这里我们可以看到这种图片的大小是 28 x 28
# print(a_data)

# 我们可以将数组展示出来，里面的 0 就表示黑色，255 表示白色
# 对于神经网络，我们第一层的输入就是 28 x 28 = 784，所以必须将得到的数据我们做一个变换，使用 reshape 将他们拉平成一个一维向量

def data_tf(x):
    x = np.array(x, dtype='float32') / 255
    x = (x - 0.5) / 0.5  # 标准化，这个技巧之后会讲到   -1~ 1
    x = x.reshape((-1,))  # 拉平
    x = torch.from_numpy(x)
    return x

train_set = mnist.MNIST('./data', train=True, transform=data_tf, download=False)  # 重新载入数据集，申明定义的数据变换
test_set = mnist.MNIST('./data', train=False, transform=data_tf, download=False)

a, a_label = train_set[0]
print(a.shape)
print(a_label)

from torch.utils.data import DataLoader
# 使用 pytorch 自带的 DataLoader 定义一个数据迭代器
train_data = DataLoader(train_set, batch_size=64, shuffle=True)     #shuffle 随机排序
test_data = DataLoader(test_set, batch_size=128, shuffle=False)

# 使用这样的数据迭代器是非常有必要的，如果数据量太大，就无法一次将他们全部读入内存，所以需要使用 python 迭代器，每次生成一个批次的数据
a, a_label = next(iter(train_data))
# 打印出一个批次的数据大小
print(a.shape)
print(a_label.shape)

# 使用 Sequential 定义 4 层神经网络
net = nn.Sequential(
    nn.Linear(784, 400),
    nn.ReLU(),
    nn.Linear(400, 200),
    nn.ReLU(),
    nn.Linear(200, 100),
    nn.ReLU(),
    nn.Linear(100, 10)
)

net

# 交叉熵在 pytorch 中已经内置了，交叉熵的数值稳定性更差，所以内置的函数已经帮我们解决了这个问题
# 定义 loss 函数
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), 1e-1)  # 使用随机梯度下降，学习率 0.1

# 开始训练
losses = []
acces = []
eval_losses = []
eval_acces = []

for e in range(10):
    train_loss = 0
    train_acc = 0
    net.train()
    for im, label in train_data:
        im = Variable(im)
        label = Variable(label)
        # 前向传播
        out = net(im)
        loss = criterion(out, label)
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # 记录误差
        train_loss +=  loss.item()
        # 计算分类的准确率
        _, pred = out.max(1)
        num_correct = (pred == label).sum().item()
        acc = num_correct / im.shape[0]
        train_acc += acc

    losses.append(int(train_loss) / len(train_data))
    acces.append(int(train_acc) / len(train_data))
    # 在测试集上检验效果
    eval_loss = 0
    eval_acc = 0
    net.eval()  # 将模型改为预测模式
    for im, label in test_data:
        im = Variable(im)
        label = Variable(label)
        out = net(im)
        loss = criterion(out, label)
        # 记录误差
        eval_loss += loss.item()
        # 记录准确率
        _, pred = out.max(1)
        num_correct = (pred == label).sum().item()
        acc = num_correct / im.shape[0]
        eval_acc += acc

    eval_losses.append( int (eval_loss) / len(test_data))
    eval_acces.append(int(eval_acc) / len(test_data))
    print('epoch: {}, Train Loss: {:.6f}, Train Acc: {:.6f}, Eval Loss: {:.6f}, Eval Acc: {:.6f}'
          .format(e, int(train_loss) / len(train_data), int(train_acc) / len(train_data), int(eval_loss) / len(test_data), int(eval_acc) / len(test_data)))

# 画出 loss 曲线和 准确率曲线

import matplotlib.pyplot as plt


plt.title('train loss')
plt.plot(np.arange(len(losses)), losses)
plt.show()

plt.plot(np.arange(len(acces)), acces)
plt.title('train acc')
plt.show()

plt.plot(np.arange(len(eval_losses)), eval_losses)
plt.title('test loss')
plt.show()

plt.plot(np.arange(len(eval_acces)), eval_acces)
plt.title('test acc')
plt.show()

# 可以看到我们的三层网络在训练集上能够达到 99.9% 的准确率，测试集上能够达到 98.20% 的准确率
