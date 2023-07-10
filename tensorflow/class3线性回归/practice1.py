import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

plotdata = {"batchsize": [], "loss": []}
def moving_average(a, w=10):
    if len(a) < w:
        return a[:]
    return [val if idx < w else sum(a[(idx - w):idx]) / w for idx, val in enumerate(a)]


train_X = np.linspace(-1,1,100)
train_Y = 2 * train_X + np.random.randn(*train_X.shape) * 0.3 #y=2x,   -1~1随机数*0.3  但是加入了噪音
#显示模拟数据点
plt.plot (train_X,train_Y,'ro',label = 'Original data')

print(train_X)
print(train_Y)

plt.legend()
plt.show()