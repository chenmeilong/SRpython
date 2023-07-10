#此代码未调试完成
# 变分编码器是自动编码器的升级版本，其结构跟自动编码器是类似的，也由编码器和解码器构成。
#
# 回忆一下，自动编码器有个问题，就是并不能任意生成图片，因为我们没有办法自己去构造隐藏向量，需要通过一张图片输入编码我们才知道得
# 到的隐含向量是什么，这时我们就可以通过变分自动编码器来解决这个问题。
#
# 其实原理特别简单，只需要在编码过程给它增加一些限制，迫使其生成的隐含向量能够粗略的遵循一个标准正态分布，这就是其与一般的自动编码器最大的不同。
# 这样我们生成一张新图片就很简单了，我们只需要给它一个标准正态分布的随机隐含向量，这样通过解码器就能够生成我们想要的图片，而不需要给它一张原始图片先编码。
# 一般来讲，我们通过 encoder 得到的隐含向量并不是一个标准的正态分布，为了衡量两种分布的相似程度，我们使用 KL divergence，利用
# 其来表示隐含向量与标准正态分布之间差异的 loss，另外一个 loss 仍然使用生成图片与原图片的均方误差来表示。
#
# ## 重参数
# 为了避免计算 KL divergence 中的积分，我们使用重参数的技巧，不是每次产生一个隐含向量，而是生成两个向量，一个表示均值，一个表示
# 标准差，这里我们默认编码之后的隐含向量服从一个正态分布的之后，就可以用一个标准正态分布先乘上标准差再加上均值来合成这个正态分布，
# 最后 loss 就是希望这个生成的正态分布能够符合一个标准正态分布，也就是希望均值为 0，方差为 1
#
import os

import torch
from torch.autograd import Variable
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

from torchvision.datasets import MNIST
from torchvision import transforms as tfs
from torchvision.utils import save_image
import numpy as np

def data_tf(x):
    x = np.array(x, dtype='float32') / 255
    x = (x - 0.5) / 0.5  # 数据预处理，标准化
    x = torch.from_numpy(x)
    return x


train_set = MNIST('./data', transform=data_tf)
train_data = DataLoader(train_set, batch_size=128, shuffle=True)


class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

        self.fc1 = nn.Linear(784, 400)
        self.fc21 = nn.Linear(400, 20) # mean
        self.fc22 = nn.Linear(400, 20) # var
        self.fc3 = nn.Linear(20, 400)
        self.fc4 = nn.Linear(400, 784)

    def encode(self, x):
        # print('##########')
        # print(x.shape)
        # x = nn.Flatten(start_dim=0)(x)
        # print(x.shape)
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        eps = torch.FloatTensor(std.size()).normal_()
        if torch.cuda.is_available():
            eps = Variable(eps.cuda())
        else:
            eps = Variable(eps)
        return eps.mul(std).add_(mu)

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return F.tanh(self.fc4(h3))

    def forward(self, x):
        # print(x.shape,'#')
        mu, logvar = self.encode(x) # 编码
        z = self.reparametrize(mu, logvar) # 重新参数化成正态分布
        return self.decode(z), mu, logvar # 解码，同时输出均值方差

net = VAE() # 实例化网络
if torch.cuda.is_available():
    net = net.cuda()

# x, _ = train_set[0]
# x = x.view(x.shape[0], -1)
# if torch.cuda.is_available():
#     x = x.cuda()
# x = Variable(x)
# _, mu, var = net(x)
# print(mu)

# 可以看到，对于输入，网络可以输出隐含变量的均值和方差，这里的均值方差还没有训练
#
# 下面开始训练

reconstruction_function = nn.MSELoss(size_average=False)

def loss_function(recon_x, x, mu, logvar):
    """
    recon_x: generating images
    x: origin images
    mu: latent mean
    logvar: latent log variance
    """
    MSE = reconstruction_function(recon_x, x)
    # loss = 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
    KLD = torch.sum(KLD_element).mul_(-0.5)
    # KL divergence
    return MSE + KLD

optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)

def to_img(x):
    '''
    定义一个函数将最后的结果转换回图片
    '''
    x = 0.5 * (x + 1.)
    x = x.clamp(0, 1)
    x = x.view(x.shape[0], 1, 28, 28)
    return x

for e in range(100):
    for im, _ in train_data:
        im = im.view(im.shape[0], -1)
        im = Variable(im)
        if torch.cuda.is_available():
            im = im.cuda()
        recon_im, mu, logvar = net(im)
        loss = loss_function(recon_im, im, mu, logvar) / im.shape[0] # 将 loss 平均
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if (e + 1) % 20 == 0:
        print('epoch: {}, Loss: {:.4f}'.format(e + 1, loss.data[0]))
        save = to_img(recon_im.cpu().data)
        if not os.path.exists('./vae_img'):
            os.mkdir('./vae_img')
        save_image(save, './vae_img/image_{}.png'.format(e + 1))


# 可以看看使用变分自动编码器得到的结果，可以发现效果比一般的编码器要好很多

# 我们可以输出其中的均值看看


x, _ = train_set[0]
x = x.view(x.shape[0], -1)
if torch.cuda.is_available():
    x = x.cuda()
x = Variable(x)
_, mu, _ = net(x)

print(mu)

# 变分自动编码器虽然比一般的自动编码器效果要好，而且也限制了其输出的编码 (code) 的概率分布，但是它仍然是通过直接计算生成图片和原始图片的均方误差来生成 loss，这个方式并不好，在下一章生成对抗网络中，我们会讲一讲这种方式计算 loss 的局限性，然后会介绍一种新的训练办法，就是通过生成对抗的训练方式来训练网络而不是直接比较两张图片的每个像素点的均方误差
