# 对于一张手写字体的图片，其大小是 28 * 28，我们可以将其看做是一个长为 28 的序列，每个序列的特征都是 28，也就是
#
# 这样我们解决了输入序列的问题，对于输出序列怎么办呢？其实非常简单，虽然我们的输出是一个序列，但是我们只需要保留其中一个作为
# 输出结果就可以了，这样的话肯定保留最后一个结果是最好的，因为最后一个结果有前面所有序列的信息，就像下面这样
#

import sys
import torch
from torch.autograd import Variable
from torch import nn
from torch.utils.data import DataLoader

from torchvision import transforms as tfs
from torchvision.datasets import mnist
from utils import train
import numpy as np


sys.path.append('..')

# 定义数据
def data_tf(x):
    x = np.array(x, dtype='float32') / 255
    x = (x - 0.5) / 0.5  # 数据预处理，标准化
    x = torch.from_numpy(x)
    return x

train_set = mnist.MNIST('./data', train=True, transform=data_tf, download=False)  # 重新载入数据集，申明定义的数据变换
test_set = mnist.MNIST('./data', train=False, transform=data_tf, download=False)
train_data = DataLoader(train_set, batch_size=64, shuffle=True)
test_data = DataLoader(test_set, batch_size=128, shuffle=False)

# 定义模型
class rnn_classify(nn.Module):
    def __init__(self, in_feature=28, hidden_feature=100, num_class=10, num_layers=2):
        super(rnn_classify, self).__init__()
        self.rnn = nn.LSTM(in_feature, hidden_feature, num_layers)  # 使用两层 lstm
        self.classifier = nn.Linear(hidden_feature, num_class)  # 将最后一个 rnn 的输出使用全连接得到最后的分类结果

    def forward(self, x):
        '''
        x 大小为 (batch, 1, 28, 28)，所以我们需要将其转换成 RNN 的输入形式，即 (28, batch, 28)
        '''
        x = x.squeeze()  # 去掉 (batch, 1, 28, 28) 中的 1，变成 (batch, 28, 28)
        x = x.permute(2, 0, 1)  # 将最后一维放到第一维，变成 (28, batch, 28)
        out, _ = self.rnn(x)  # 使用默认的隐藏状态，得到的 out 是 (28, batch, hidden_feature)
        out = out[-1, :, :]  # 取序列中的最后一个，大小是 (batch, hidden_feature)
        out = self.classifier(out)  # 得到分类结果
        return out

net = rnn_classify()
criterion = nn.CrossEntropyLoss()

optimzier = torch.optim.Adadelta(net.parameters(), 1e-1)

# 开始训练
train(net, train_data, test_data, 10, optimzier, criterion)

