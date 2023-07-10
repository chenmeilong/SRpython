# 这个项目包含了吴恩达机器学习ex3的python实现，主要知识点为多类别逻辑回归、神经网络，题目内容可以查看数据集中的ex3.pdf

# # 1 多分类
# 这个部分需要你实现手写数字（0到9）的识别。你需要扩展之前的逻辑回归，并将其应用于一对多的分类。

# ## 数据集
# 这是一个MATLAB格式的.m文件，其中包含5000个20*20像素的手写字体图像，以及他对应的数字。另外，数字0的y值，对应的是10
# 用Python读取我们需要使用SciPy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from scipy.io import loadmat
from sklearn.metrics import classification_report  # 这个包是评价报告

data = loadmat('ex3data1.mat')
print(data)
print(data['X'].shape, data['y'].shape)   #(5000, 400) (5000, 1)

# ## 1.2 数据可视化
# 随机展示100个数据
sample_idx = np.random.choice(np.arange(data['X'].shape[0]), 100)
sample_images = data['X'][sample_idx, :]
print(np.shape(sample_images))
print(data['y'][sample_idx, :].reshape((10, 10)))
fig, ax_array = plt.subplots(nrows=10, ncols=10, sharey=True, sharex=True, figsize=(12, 12))
for r in range(10):
    for c in range(10):
        ax_array[r, c].matshow(np.array(sample_images[10 * r + c].reshape((20, 20))).T, cmap=matplotlib.cm.binary)
        plt.xticks(np.array([]))
        plt.yticks(np.array([]))
plt.show()

# ## 1.3 将逻辑回归向量化
# 你将用多分类逻辑回归做一个分类器。因为现在有10个数字类别，所以你需要训练10个不同的逻辑回归分类器。为了让训练效率更高，将逻辑回归向量化是非常重要的，不要用循环。
# ### 1.3.1 向量化代价函数
# # sigmoid 函数
def sigmoid(z):
    return 1 / (1 + np.exp(-z))
# 带正则化的代价函数：
def cost(theta, X, y, learningRate):
    theta = np.matrix(theta)
    X = np.matrix(X)
    y = np.matrix(y)
    first = np.multiply(-y, np.log(sigmoid(X * theta.T)))
    second = np.multiply((1 - y), np.log(1 - sigmoid(X * theta.T)))
    reg = (learningRate / (2 * len(X))) * np.sum(np.power(theta[:, 1:theta.shape[1]], 2))#从1开始求和
    return np.sum(first - second) / len(X) + reg

# ### 1.3.3 向量化正则化逻辑回归
# 向量化后的梯度更新公式如下：
def gradient(theta, X, y, learningRate):
    theta = np.matrix(theta)
    X = np.matrix(X)
    y = np.matrix(y)
    parameters = int(theta.ravel().shape[1])  #参数401
    error = sigmoid(X * theta.T) - y
    grad = ((X.T * error) / len(X)).T + ((learningRate / len(X)) * theta)
    # θ(0)个参数不用正则化
    grad[0, 0] = np.sum(np.multiply(error, X[:, 0])) / len(X)
    return np.array(grad).ravel()

# ## 1.4 一对多分类器
# 现在我们已经定义了代价函数和梯度函数，现在是构建分类器的时候了。
# 对于这个任务，我们有10个可能的类，并且由于逻辑回归只能一次在2个类之间进行分类，我们需要多类分类的策略。
# 在本练习中，我们的任务是实现一对一全分类方法，其中具有k个不同类的标签就有k个分类器，每个分类器在“类别 i”和“不是 i”之间决定。
# 我们将把分类器训练包含在一个函数中，该函数计算10个分类器中的每个分类器的最终权重，并将权重返回为k*(n + 1)数组，其中n是参数数量。
from scipy.optimize import minimize

def one_vs_all(X, y, num_labels, learning_rate):
    rows = X.shape[0]
    params = X.shape[1]
    # k * (n + 1) array for the parameters of each of the k classifiers
    all_theta = np.zeros((num_labels, params + 1))  #[10,401]
    # insert a column of ones at the beginning for the intercept term
    X = np.insert(X, 0, values=np.ones(rows), axis=1) #[5000,401]
    # labels are 1-indexed instead of 0-indexed
    for i in range(1, num_labels + 1):     #i取1-10    等效成10个2分类问题，训练10个分类器
        theta = np.zeros(params + 1)      #401
        y_i = np.array([1 if label == i else 0 for label in y])
        y_i = np.reshape(y_i, (rows, 1))  #[5000,1]
        # minimize the objective function
        fmin = minimize(fun=cost, x0=theta, args=(X, y_i, learning_rate), method='TNC', jac=gradient)
        all_theta[i - 1, :] = fmin.x
    return all_theta

# 这里需要注意的几点：首先，我们为theta添加了一个额外的参数（与训练数据一列），以计算截距项（常数项）。
# 其次，我们将y从类标签转换为每个分类器的二进制值（要么是类i，要么不是类i）。
# 最后，我们使用SciPy的较新优化API来最小化每个分类器的代价函数。
# 如果指定的话，API将采用目标函数，初始参数集，优化方法和jacobian（渐变）函数。 然后将优化程序找到的参数分配给参数数组。
# 实现向量化代码的一个更具挑战性的部分是正确地写入所有的矩阵，保证维度正确。
rows = data['X'].shape[0]
params = data['X'].shape[1]
all_theta = np.zeros((10, params + 1))
X = np.insert(data['X'], 0, values=np.ones(rows), axis=1)
theta = np.zeros(params + 1)
y_0 = np.array([1 if label == 0 else 0 for label in data['y']])
y_0 = np.reshape(y_0, (rows, 1))
print(X.shape, y_0.shape, theta.shape, all_theta.shape)  #(5000, 401) (5000, 1) (401,) (10, 401)

# 注意，theta是一维数组，因此当它被转换为计算梯度的代码中的矩阵时，它变为（1×401）矩阵。 我们还检查y中的类标签，以确保它们看起来像我们想象的一致。
print(np.unique(data['y']))  # 看下有几类标签  [ 1  2  3  4  5  6  7  8  9 10]   10表示0
# 让我们确保我们的训练函数正确运行，并且得到合理的输出。
all_theta = one_vs_all(data['X'], data['y'], 10, 1)
print(all_theta.shape)   #(10, 401)

# ### 1.4.1 一对多预测
# 我们现在准备好最后一步 - 使用训练完毕的分类器预测每个图像的标签。
# 对于这一步，我们将计算每个类的类概率，对于每个训练样本（使用当然的向量化代码），并将输出类标签为具有最高概率的类。
def predict_all(X, all_theta):
    rows = X.shape[0]
    params = X.shape[1]
    num_labels = all_theta.shape[0]
    # same as before, insert ones to match the shape
    X = np.insert(X, 0, values=np.ones(rows), axis=1)
    # convert to matrices
    X = np.matrix(X)
    all_theta = np.matrix(all_theta)
    # compute the class probability for each class on each training instance
    h = sigmoid(X * all_theta.T)
    # create array of the index with the maximum probability
    h_argmax = np.argmax(h, axis=1)
    # because our array was zero-indexed we need to add one for the true label prediction
    h_argmax = h_argmax + 1
    return h_argmax

# 现在我们可以使用predict_all函数为每个实例生成类预测，看看我们的分类器是如何工作的。
y_pred = predict_all(data['X'], all_theta)
print(classification_report(data['y'], y_pred))

# # 2 神经网络
# 在前面一个部分，我们已经实现了多分类逻辑回归来识别手写数字。但是，逻辑回归并不能承载更复杂的假设，因为他就是个线性分类器。
# 这部分，你需要实现一个可以识别手写数字的神经网络。神经网络可以表示一些非线性复杂的模型。权重已经预先训练好，你的目标是在现有权重基础上，实现前馈神经网络。
# ## 2.1 模型表达

# 输入是图片的像素值，20*20像素的图片有400个输入层单元，不包括需要额外添加的加上常数项。
# 材料已经提供了训练好的神经网络的参数  400个输入单元有25个隐层单元和10个输出单元（10个输出）

# ## 2.2 前馈神经网络和预测
# 你需要实现前馈神经网络预测手写数字的功能。和之前的一对多分类一样，神经网络的预测会把输出单元中值最大的，作为预测输出

weight = loadmat("ex3weights.mat")
theta1, theta2 = weight['Theta1'], weight['Theta2']
print(theta1.shape, theta2.shape)  #(25, 401) (10, 26)
# 插入常数项
X2 = np.matrix(np.insert(data['X'], 0, values=np.ones(X.shape[0]), axis=1))
y2 = np.matrix(data['y'])
print(X2.shape, y2.shape)   #(5000, 401) (5000, 1)

a1 = X2
z2 = a1 * theta1.T
print('z2shape:',z2.shape)
a2 = sigmoid(z2)
print('a2shape:',a2.shape)
a2 = np.insert(a2, 0, values=np.ones(a2.shape[0]), axis=1)
z3 = a2 * theta2.T
print('z3shape:',z3.shape)
a3 = sigmoid(z3)
print('a3shape:',a3.shape)

y_pred2 = np.argmax(a3, axis=1) + 1
print(y_pred2.shape)
print(classification_report(y2, y_pred))