
import numpy as np
import torch
from torchvision.datasets import mnist  # 导入 pytorch 内置的 mnist 数据
from torch.utils.data import DataLoader
from torch import nn
from torch.autograd import Variable
from utils import train


def data_tf(x):
    x = np.array(x, dtype='float32') / 255
    x = (x - 0.5) / 0.5  # 数据预处理，标准化
    x = torch.from_numpy(x)
    x = x.unsqueeze(0)
    return x

train_set = mnist.MNIST('./data', train=True, transform=data_tf, download=False)  # 重新载入数据集，申明定义的数据变换
test_set = mnist.MNIST('./data', train=False, transform=data_tf, download=False)
train_data = DataLoader(train_set, batch_size=640, shuffle=True)
test_data = DataLoader(test_set, batch_size=128, shuffle=False)

# 使用批标准化
class conv_bn_net(nn.Module):
    def __init__(self):
        super(conv_bn_net, self).__init__()
        self.stage1 = nn.Sequential(
            nn.Conv2d(1, 6, 3, padding=1),
            nn.BatchNorm2d(6),    #批量标准化
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(6, 16, 5),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2)
        )

        self.classfy = nn.Linear(400, 10)

    def forward(self, x):
        x = self.stage1(x)
        print(x.shape)
        x = x.view(x.shape[0], -1)
        print(x.shape)
        x = self.classfy(x)
        return x


criterion = nn.CrossEntropyLoss()

net = conv_bn_net()
optimizer = torch.optim.SGD(net.parameters(), 1e-1)  # 使用随机梯度下降，学习率 0.1

train(net, train_data, test_data, 5, optimizer, criterion)  # 5是迭代次数








