# 这个项目包含了吴恩达机器学习ex8的python实现，主要知识点为异常检测和推荐系统，题目内容可以查看数据集中的ex8.pdf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from scipy.io import loadmat
# 在本练习中，我们将使用高斯模型实现异常检测算法，并将其应用于检测网络上的故障服务器。 我们还将看到如何使用协作过滤构建推荐系统，并将其应用于电影推荐数据集。

# # 1 Anomaly detection（异常检测）
# 我们的第一个任务是使用高斯模型来检测数据集中未标记的示例是否应被视为异常。 我们先从简单的二维数据集开始。

# **数据可视化**
data = loadmat('ex8data1.mat')
X = data['X']
print(X.shape)
fig, ax = plt.subplots(figsize=(12, 8))
ax.scatter(X[:, 0], X[:, 1])
plt.show()

# ## 1.1 高斯分布
# 我们需要为每个特征$x_i$拟合一个高斯分布，并返回高斯分布的参数$\mu_i,\sigma_i^2$。高斯分布公式如下：
# 其中，$\mu$是平均值，$\sigma^2$是方差

# ## 1.2 计算高斯分布参数
# 你要做的是，输入一个X矩阵，输出2个n维的向量，mu包含了每一个维度的平均值，sigma2包含了每一个维度的方差。

def estimate_gaussian(X):
    mu = X.mean(axis=0)
    sigma = X.var(axis=0)    #方差var
    return mu, sigma
mu, sigma = estimate_gaussian(X)
print(mu, sigma)   #[14.11222578 14.99771051] [1.83263141 1.70974533]

# **数据可视化**
xplot = np.linspace(0, 25, 100)
yplot = np.linspace(0, 25, 100)
Xplot, Yplot = np.meshgrid(xplot, yplot)
Z = np.exp((-0.5) * ((Xplot - mu[0]) ** 2 / sigma[0] + (Yplot - mu[1]) ** 2 / sigma[1]))

fig, ax = plt.subplots(figsize=(12, 8))
contour = plt.contour(Xplot, Yplot, Z, [10 ** -11, 10 ** -7, 10 ** -5, 10 ** -3, 0.1], colors='k')
ax.scatter(X[:, 0], X[:, 1])
plt.show()

# ## 1.3 选择阈值ε
# 有了参数后，可以估计每组数据的概率，低概率的数据点更可能是异常的。确定异常点需要先确定一个阈值，我们可以通过验证集集来确定这个阈值。
Xval = data['Xval']
yval = data['yval']
print(Xval.shape, yval.shape)         #(307, 2) (307, 1)

# 我们还需要一种计算数据点属于正态分布的概率的方法。 幸运的是SciPy有这个内置的方法。
from scipy import stats
dist = stats.norm(mu[0], sigma[0])
print(dist.pdf(15))                  #0.1935875044615038

# 我们还可以将数组传递给概率密度函数，并获得数据集中每个点的概率密度。
print(dist.pdf(X[:, 0])[0:50])

# 我们计算并保存给定上述的高斯模型参数的数据集中每个值的概率密度。
p = np.zeros((X.shape[0], X.shape[1]))
p[:, 0] = stats.norm(mu[0], sigma[0]).pdf(X[:, 0])
p[:, 1] = stats.norm(mu[1], sigma[1]).pdf(X[:, 1])
print(p.shape)   #(307, 2)   转换成正太分布

# 我们还需要为验证集（使用相同的模型参数）执行此操作。 我们将使用与真实标签组合的这些概率来确定将数据点分配为异常的最佳概率阈值。
pval = np.zeros((Xval.shape[0], Xval.shape[1]))
pval[:, 0] = stats.norm(mu[0], sigma[0]).pdf(Xval[:, 0])
pval[:, 1] = stats.norm(mu[1], sigma[1]).pdf(Xval[:, 1])
print(pval.shape)   #(307, 2)

# 接下来，我们需要一个函数，找到给定概率密度值和真实标签的最佳阈值。 为了做到这一点，我们将为不同的epsilon值计算F1分数。
# F1是真阳性，假阳性和假阴性的数量的函数。 方程式在练习文本中。
def select_threshold(pval, yval):
    best_epsilon = 0
    best_f1 = 0
    f1 = 0
    step = (pval.max() - pval.min()) / 1000
    for epsilon in np.arange(pval.min(), pval.max(), step):
        preds = pval < epsilon
        tp = np.sum(np.logical_and(preds == 1, yval == 1)).astype(float)
        fp = np.sum(np.logical_and(preds == 1, yval == 0)).astype(float)
        fn = np.sum(np.logical_and(preds == 0, yval == 1)).astype(float)
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1 = (2 * precision * recall) / (precision + recall)
        if f1 > best_f1:
            best_f1 = f1
            best_epsilon = epsilon
    return best_epsilon, best_f1

epsilon, f1 = select_threshold(pval, yval)
print(epsilon, f1)      #0.009566706005956842 0.7142857142857143

# 最后，我们可以将阈值应用于数据集，并可视化结果。

# indexes of the values considered to be outliers
outliers = np.where(p < epsilon)
print(outliers)   #(array([300, 301, 301, 303, 303, 304, 306, 306], dtype=int64), array([1, 0, 1, 0, 1, 0, 0, 1], dtype=int64))

fig, ax = plt.subplots(figsize=(12, 8))
ax.scatter(X[:, 0], X[:, 1])
ax.scatter(X[outliers[0], 0], X[outliers[0], 1], s=50, color='r', marker='o')
plt.show()
# 红点是被标记为异常值的点。 这些看起来很合理。 有一些分离（但没有被标记）的右上角也可能是一个异常值，但是相当接近。

