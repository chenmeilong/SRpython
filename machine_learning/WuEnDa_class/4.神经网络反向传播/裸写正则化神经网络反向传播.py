# 这个项目包含了吴恩达机器学习ex4的python实现，主要知识点为反向传播神经网络，题目内容可以查看数据集中的ex4.pdf

# # 1 神经网络
# 对于这个练习，我们将再次处理手写数字数据集。这次使用反向传播的前馈神经网络，自动学习神经网络的参数。
# ## 1.1 数据可视化
# 这部分和ex3里是一样的，5000张20*20像素的手写数字数据集，以及对应的数字（1-9，0对应10）
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from scipy.io import loadmat
from sklearn.preprocessing import OneHotEncoder

data = loadmat('ex4data1.mat')
X = data['X']
y = data['y']
print(X.shape, y.shape)  # 看下维度  (5000, 400) (5000, 1)

weight = loadmat("ex4weights.mat")
theta1, theta2 = weight['Theta1'], weight['Theta2']
print(theta1.shape, theta2.shape)    #(25, 401) (10, 26)

sample_idx = np.random.choice(np.arange(data['X'].shape[0]), 100)
sample_images = data['X'][sample_idx, :]
fig, ax_array = plt.subplots(nrows=10, ncols=10, sharey=True, sharex=True, figsize=(12, 12))
for r in range(10):
    for c in range(10):
        ax_array[r, c].matshow(np.array(sample_images[10 * r + c].reshape((20, 20))).T, cmap=matplotlib.cm.binary)
        plt.xticks(np.array([]))
        plt.yticks(np.array([]))
plt.show()

# ## 1.2 模型展示
# 这部分和ex3的第二部分一样

# ## 1.3 前向传播和代价函数
# 首先，实现神经网络的代价函数和梯度函数
# 要求：你的代码应该适用于任何数据集，包括任意数量的输入输出单元
def sigmoid(z):
    return 1 / (1 + np.exp(-z))
# 前向传播函数
def forward_propagate(X, theta1, theta2):
    m = X.shape[0]
    a1 = np.insert(X, 0, values=np.ones(m), axis=1)
    z2 = a1 * theta1.T
    a2 = np.insert(sigmoid(z2), 0, values=np.ones(m), axis=1)
    z3 = a2 * theta2.T
    h = sigmoid(z3)
    return a1, z2, a2, z3, h

# 代价函数
def cost(theta1, theta2, input_size, hidden_size, num_labels, X, y, learning_rate):
    m = X.shape[0]    #5000
    X = np.matrix(X)
    y = np.matrix(y)
    # run the feed-forward pass
    a1, z2, a2, z3, h = forward_propagate(X, theta1, theta2)
    # compute the cost
    J = 0
    for i in range(m):
        first_term = np.multiply(-y[i, :], np.log(h[i, :]))
        second_term = np.multiply((1 - y[i, :]), np.log(1 - h[i, :]))
        J += np.sum(first_term - second_term)
    J = J / m
    return J

# 对y标签进行编码
# 一开始我们得到的y是$5000*1$维的向量，但我们要把他编码成$5000*10$的矩阵。
# 比如说，原始$y_0=2$，那么转化后的Y对应行就是[0,1,0...0]，原始$y_1=0$转化后的Y对应行就是[0,0...0,1]
#
# Scikitlearn有一个内置的编码函数，我们可以使用这个。
encoder = OneHotEncoder(sparse=False)
y_onehot = encoder.fit_transform(y)
print(y_onehot.shape)   #(5000, 10)
print(y[0], y_onehot[0, :])  #[10] [0. 0. 0. 0. 0. 0. 0. 0. 0. 1.]

# 用ex4weights.mat中给定的theta1 和 theta2 计算初始代价
# 初始化设置
input_size = 400
hidden_size = 25
num_labels = 10
learning_rate = 1
print(cost(theta1, theta2, input_size, hidden_size, num_labels, X, y_onehot, learning_rate))  # 答案应该是0.287629

# ## 1.4 正则化代价函数
# 公式如下：
# 用ex4weights.mat中给定的theta1 和 theta2 计算初始代价
def costReg(theta1, theta2, input_size, hidden_size, num_labels, X, y, learning_rate):
    m = X.shape[0]
    X = np.matrix(X)
    y = np.matrix(y)
    # run the feed-forward pass
    a1, z2, a2, z3, h = forward_propagate(X, theta1, theta2)
    # compute the cost
    J = 0
    for i in range(m):
        first_term = np.multiply(-y[i, :], np.log(h[i, :]))
        second_term = np.multiply((1 - y[i, :]), np.log(1 - h[i, :]))
        J += np.sum(first_term - second_term)

    J = J / m
    # add the cost regularization term
    J += (float(learning_rate) / (2 * m)) * (np.sum(np.power(theta1[:, 1:], 2)) + np.sum(np.power(theta2[:, 1:], 2)))
    return J
print(costReg(theta1, theta2, input_size, hidden_size, num_labels, X, y_onehot, learning_rate))   # 答案应该是0.383770


# # 2 反向传播
# 这一部分需要你实现反向传播的算法，来计算神经网络代价函数的梯度。获得了梯度的数据，我们就可以使用工具库来计算代价函数的最小值。
# ## 2.1 sigmoid梯度
# 在绝对值比较大的数上，梯度应该接近0，当$z=0$时，梯度应该是0.25
# 另外，这个函数应该可以作用于向量以及矩阵，作用在矩阵上时，应该是计算每个元素的梯度
def sigmoid_gradient(z):
    return np.multiply(sigmoid(z), (1 - sigmoid(z)))
print(sigmoid_gradient(0))

# ## 2.2 随机初始
# 当我们训练神经网络的时候，需要将θ^{(l)}$设定为$\{-\epsilon _{init},\epsilon _{init}\}$之间的随机值。此处我们设定$\epsilon _{init}=0.12$
# 这个范围保证了参数足够小，使参数学习更高效
# np.random.random(size) 返回size大小的0-1随机浮点数
params = (np.random.random(size=hidden_size * (input_size + 1) + num_labels * (hidden_size + 1)) - 0.5) * 0.24   #-1~1的随机数

# ## 2.3 反向传播
# 反向传播的步骤是，给定训练集$(x^{(t},y^{(t)})$，先计算正向传播$h_\Theta(x)$，再对于$l$层的每个节点$j$，
# 计算误差项$\delta_j^{(l)}$，这个数据衡量这个节点对最后输出的误差“贡献”了多少。
# 对于每个输出节点，我们可以直接计算输出值与目标值的差值，定义为$\delta_j^{(3)}$。对于每个隐藏节点，需要基于现有权重及$(l+1)$层的误差，计算$\delta_j^{(l)}$

def backprop(params, input_size, hidden_size, num_labels, X, y, learning_rate):
    m = X.shape[0]
    X = np.matrix(X)
    y = np.matrix(y)
    # reshape the parameter array into parameter matrices for each layer
    theta1 = np.matrix(np.reshape(params[:hidden_size * (input_size + 1)], (hidden_size, (input_size + 1))))
    theta2 = np.matrix(np.reshape(params[hidden_size * (input_size + 1):], (num_labels, (hidden_size + 1))))
    # run the feed-forward pass
    a1, z2, a2, z3, h = forward_propagate(X, theta1, theta2)
    # initializations
    J = 0
    delta1 = np.zeros(theta1.shape)  # (25, 401)
    delta2 = np.zeros(theta2.shape)  # (10, 26)
    # compute the cost
    for i in range(m):
        first_term = np.multiply(-y[i, :], np.log(h[i, :]))
        second_term = np.multiply((1 - y[i, :]), np.log(1 - h[i, :]))
        J += np.sum(first_term - second_term)
    J = J / m
    # perform backpropagation
    for t in range(m):
        a1t = a1[t, :]  # (1, 401)  第t个样本的a1
        z2t = z2[t, :]  # (1, 25)
        a2t = a2[t, :]  # (1, 26)
        ht = h[t, :]  # (1, 10)
        yt = y[t, :]  # (1, 10)
        d3t = ht - yt  # (1, 10)   理解成第三层的梯度误差    后面要往前推导
        z2t = np.insert(z2t, 0, values=np.ones(1))  # (1, 26)
        d2t = np.multiply((theta2.T * d3t.T).T, sigmoid_gradient(z2t))  # (1, 26)
        delta1 = delta1 + (d2t[:, 1:]).T * a1t           #更新 梯度下降更新权重的值
        delta2 = delta2 + d3t.T * a2t
    delta1 = delta1 / m
    delta2 = delta2 / m
    return J, delta1, delta2


# ## 2.4 梯度校验
# 进行梯度校验是，你需要把θ^{(1)},\Theta^{(2)}$连接成一个长向量$\theta$。之后你可以使用如下公式计算$\frac{\partial}{\partial \theta _i}j(\theta)$:
# 如果你的反向传播计算正确，那你得出的这个数字应该小于10e-9
# 运行一次巨慢，不做了

# ## 2.5 正则化神经网络
# 加入正则项
def backpropReg(params, input_size, hidden_size, num_labels, X, y, learning_rate):
    m = X.shape[0]
    X = np.matrix(X)
    y = np.matrix(y)
    # reshape the parameter array into parameter matrices for each layer
    theta1 = np.matrix(np.reshape(params[:hidden_size * (input_size + 1)], (hidden_size, (input_size + 1))))
    theta2 = np.matrix(np.reshape(params[hidden_size * (input_size + 1):], (num_labels, (hidden_size + 1))))
    # run the feed-forward pass
    a1, z2, a2, z3, h = forward_propagate(X, theta1, theta2)
    # initializations
    J = 0
    delta1 = np.zeros(theta1.shape)  # (25, 401)
    delta2 = np.zeros(theta2.shape)  # (10, 26)
    # compute the cost
    for i in range(m):
        first_term = np.multiply(-y[i, :], np.log(h[i, :]))
        second_term = np.multiply((1 - y[i, :]), np.log(1 - h[i, :]))
        J += np.sum(first_term - second_term)
    J = J / m
    # add the cost regularization term
    J += (float(learning_rate) / (2 * m)) * (np.sum(np.power(theta1[:, 1:], 2)) + np.sum(np.power(theta2[:, 1:], 2)))
    # perform backpropagation
    for t in range(m):
        a1t = a1[t, :]  # (1, 401)
        z2t = z2[t, :]  # (1, 25)
        a2t = a2[t, :]  # (1, 26)
        ht = h[t, :]  # (1, 10)
        yt = y[t, :]  # (1, 10)
        d3t = ht - yt  # (1, 10)
        z2t = np.insert(z2t, 0, values=np.ones(1))  # (1, 26)
        d2t = np.multiply((theta2.T * d3t.T).T, sigmoid_gradient(z2t))  # (1, 26)
        delta1 = delta1 + (d2t[:, 1:]).T * a1t
        delta2 = delta2 + d3t.T * a2t
    delta1 = delta1 / m
    delta2 = delta2 / m
    # add the gradient regularization term  正则化
    delta1[:, 1:] = delta1[:, 1:] + (theta1[:, 1:] * learning_rate) / m
    delta2[:, 1:] = delta2[:, 1:] + (theta2[:, 1:] * learning_rate) / m
    # unravel the gradient matrices into a single array
    grad = np.concatenate((np.ravel(delta1), np.ravel(delta2)))
    return J, grad


# ## 2.6 使用工具库计算参数最优解
from scipy.optimize import minimize
# minimize the objective function
fmin = minimize(fun=backpropReg, x0=(params), args=(input_size, hidden_size, num_labels, X, y_onehot, learning_rate),
                method='TNC', jac=True, options={'maxiter': 250})
print(fmin)


X = np.matrix(X)
thetafinal1 = np.matrix(np.reshape(fmin.x[:hidden_size * (input_size + 1)], (hidden_size, (input_size + 1))))
thetafinal2 = np.matrix(np.reshape(fmin.x[hidden_size * (input_size + 1):], (num_labels, (hidden_size + 1))))
# 计算使用优化后的θ得出的预测
a1, z2, a2, z3, h = forward_propagate(X, thetafinal1, thetafinal2 )
y_pred = np.array(np.argmax(h, axis=1) + 1)
print(y_pred)


# 最后，我们可以计算准确度，看看我们训练完毕的神经网络效果怎么样。
# 预测值与实际值比较
from sklearn.metrics import classification_report#这个包是评价报告
print(classification_report(y, y_pred))

# # 3 可视化隐藏层
hidden_layer = thetafinal1[:, 1:]
print(hidden_layer.shape)         #(25, 400)

fig, ax_array = plt.subplots(nrows=5, ncols=5, sharey=True, sharex=True, figsize=(12, 12))
for r in range(5):
    for c in range(5):
        ax_array[r, c].matshow(np.array(hidden_layer[5 * r + c].reshape((20, 20))),cmap=matplotlib.cm.binary)
        plt.xticks(np.array([]))
        plt.yticks(np.array([]))
plt.show()

