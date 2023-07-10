# 这个项目包含了吴恩达机器学习ex5的python实现，主要知识点为偏差和方差，训练集&验证集&测试集，题目内容可以查看数据集中的ex5.pdf

import numpy as np
import scipy.io as sio
import scipy.optimize as opt
import pandas as pd
import matplotlib.pyplot as plt
#import seaborn as sns


# # 1 正则化线性回归
# 这一部分，我们需要先对一个水库的流出水量以及水库水位进行正则化线性归回。然后将会探讨方差-偏差的问题

# ## 1.1 数据可视化
data = sio.loadmat('ex5data1.mat')
X, y, Xval, yval, Xtest, ytest = map(np.ravel,[data['X'], data['y'], data['Xval'], data['yval'], data['Xtest'], data['ytest']])
print(X.shape, y.shape, Xval.shape, yval.shape, Xtest.shape, ytest.shape)   #(12,) (12,) (21,) (21,) (21,) (21,)

fig, ax = plt.subplots(figsize=(12, 8))
ax.scatter(X, y)
ax.set_xlabel('water_level')
ax.set_ylabel('flow')
plt.show()

# ## 1.2 正则化线性回归代价函数
X, Xval, Xtest = [np.insert(x.reshape(x.shape[0], 1), 0, np.ones(x.shape[0]), axis=1) for x in (X, Xval, Xtest)]
def cost(theta, X, y):
    m = X.shape[0]
    inner = X @ theta - y  # R(m*1)
    square_sum = inner.T @ inner
    cost = square_sum / (2 * m)
    return cost

def costReg(theta, X, y, reg=1):
    m = X.shape[0]          #12
    regularized_term = (reg / (2 * m)) * np.power(theta[1:], 2).sum()
    return cost(theta, X, y) + regularized_term

theta = np.ones(X.shape[1])  #  X.shape[1] 2
print(costReg(theta, X, y, 1))  #303.9931922202643


# ## 1.3 正则化线性回归的梯度
# 设定$\theta$初始值为[1,1]，输出应该为[-15.30, 598.250]
def gradient(theta, X, y):
    m = X.shape[0]
    inner = X.T @ (X @ theta - y)  # (m,n).T @ (m, 1) -> (n, 1)
    return inner / m
def gradientReg(theta, X, y, reg):
    m = X.shape[0]
    regularized_term = theta.copy()  # same shape as theta
    regularized_term[0] = 0  # don't regularize intercept theta
    regularized_term = (reg / m) * regularized_term
    return gradient(theta, X, y) + regularized_term

print(gradientReg(theta, X, y, 1))   #[-15.30301567 598.25074417]

# ## 1.4 拟合线性回归
# 调用工具库找到$\theta$最优解，在这个部分，我们令$\lambda=0$。因为我们现在训练的是2维的$\theta$，所以正则化不会对这种低维的$\theta$有很大的帮助。
# 完成之后，将数据和拟合曲线可视化。
theta = np.ones(X.shape[1])
final_theta = opt.minimize(fun=costReg, x0=theta, args=(X, y, 0), method='TNC', jac=gradientReg,options={'disp': True}).x
print(final_theta)   #[13.08790348  0.36777923]

b = final_theta[0]  # intercept
m = final_theta[1]  # slope

fig, ax = plt.subplots(figsize=(12, 8))
plt.scatter(X[:, 1], y, c='r', label="Training data")
plt.plot(X[:, 1], X[:, 1] * m + b, c='b', label="Prediction")
ax.set_xlabel('water_level')
ax.set_ylabel('flow')
ax.legend()
plt.show()

# # 2 方差和偏差
# 机器学习中的一个重要概念是偏差-方差权衡。偏差较大的模型会欠拟合，而方差较大的模型会过拟合。这部分会让你画出学习曲线来判断方差和偏差的问题。

# ## 2.1 学习曲线
# 1.使用训练集的子集来拟合应模型
# 2.在计算训练代价和验证集代价时，没有用正则化
# 3.记住使用相同的训练集子集来计算训练代价
def linear_regression(X, y, l=1):
    # init theta
    theta = np.ones(X.shape[1])
    # train it
    res = opt.minimize(fun=costReg,
                       x0=theta,
                       args=(X, y, l),
                       method='TNC',
                       jac=gradientReg,
                       options={'disp': True})
    return res

training_cost, cv_cost = [], []
m = X.shape[0] #12
for i in range(1, m + 1):
    res = linear_regression(X[:i, :], y[:i], 0)
    tc = costReg(res.x, X[:i, :], y[:i], 0)
    cv = costReg(res.x, Xval, yval, 0)
    training_cost.append(tc)
    cv_cost.append(cv)

fig, ax = plt.subplots(figsize=(12, 8))
plt.plot(np.arange(1, m + 1), training_cost, label='training cost')
plt.plot(np.arange(1, m + 1), cv_cost, label='cv cost')
plt.legend()
plt.show()
# 这个模型拟合不太好, **欠拟合了**

# # 3 多项式回归
# 线性回归对于现有数据来说太简单了，会欠拟合，我们需要多添加一些特征。

# 写一个函数，输入原始X，和幂的次数p，返回X的1到p次幂
def poly_features(x, power, as_ndarray=False):
    data = {'f{}'.format(i): np.power(x, i) for i in range(1, power + 1)}
    df = pd.DataFrame(data)
    return df.values if as_ndarray else df

data = sio.loadmat('ex5data1.mat')
X, y, Xval, yval, Xtest, ytest = map(np.ravel,
                                     [data['X'], data['y'], data['Xval'], data['yval'], data['Xtest'], data['ytest']])
poly_features(X, power=3)

# # 3.1 多项式回归
# 1. 使用之前的代价函数和梯度函数
# 2. 扩展特征到8阶特征
# 3. 使用 **归一化** 来处理 $x^n$
# 4. $\lambda=0$
def normalize_feature(df):
    """Applies function along input axis(default 0) of DataFrame."""
    return df.apply(lambda column: (column - column.mean()) / column.std())
def prepare_poly_data(*args, power):
    def prepare(x):
        # expand feature
        df = poly_features(x, power=power)
        # normalization
        ndarr = normalize_feature(df).values
        # add intercept term
        return np.insert(ndarr, 0, np.ones(ndarr.shape[0]), axis=1)
    return [prepare(x) for x in args]

X_poly, Xval_poly, Xtest_poly = prepare_poly_data(X, Xval, Xtest, power=8)
print(X_poly[:3, :])

# 画出学习曲线
# 首先，我们没有使用正则化，所以 $\lambda=0$
def plot_learning_curve(X, Xinit, y, Xval, yval, l=0):
    training_cost, cv_cost = [], []
    m = X.shape[0]
    for i in range(1, m + 1):
        # regularization applies here for fitting parameters
        res = linear_regression(X[:i, :], y[:i], l=l)
        # remember, when you compute the cost here, you are computing
        # non-regularized cost. Regularization is used to fit parameters only
        tc = cost(res.x, X[:i, :], y[:i])
        cv = cost(res.x, Xval, yval)
        training_cost.append(tc)
        cv_cost.append(cv)
    fig, ax = plt.subplots(2, 1, figsize=(12, 12))
    ax[0].plot(np.arange(1, m + 1), training_cost, label='training cost')
    ax[0].plot(np.arange(1, m + 1), cv_cost, label='cv cost')
    ax[0].legend()

    fitx = np.linspace(-50, 50, 100)
    fitxtmp = prepare_poly_data(fitx, power=8)
    fity = np.dot(prepare_poly_data(fitx, power=8)[0], linear_regression(X, y, l).x.T)

    ax[1].plot(fitx, fity, c='r', label='fitcurve')
    ax[1].scatter(Xinit, y, c='b', label='initial_Xy')

    ax[1].set_xlabel('water_level')
    ax[1].set_ylabel('flow')

plot_learning_curve(X_poly, X, y, Xval_poly, yval, l=0)
plt.show()
# 你可以看到训练的代价太低了，不真实. 这是 **过拟合**了

# ## 3.2 调整正则化系数λ
plot_learning_curve(X_poly, X, y, Xval_poly, yval, l=1)
plt.show()
# 训练代价不再是0了，也就是说我们减轻**过拟合**
plot_learning_curve(X_poly, X, y, Xval_poly, yval, l=100)
plt.show()
# 太多正则化惩罚太多，变成 **欠拟合**状态


# ## 3.3 找到最佳的λ
# 通过之前的实验，我们可以发现$\lambda$可以极大程度地影响正则化多项式回归。
# 所以这部分我们会会使用验证集去评价$\lambda$的表现好坏，然后选择表现最好的$\lambda$后，用测试集测试模型在没有出现过的数据上会表现多好。
# 尝试$\lambda$值[0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10]
l_candidate = [0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10]
training_cost, cv_cost = [], []
for l in l_candidate:
    res = linear_regression(X_poly, y, l)
    tc = cost(res.x, X_poly, y)
    cv = cost(res.x, Xval_poly, yval)
    training_cost.append(tc)
    cv_cost.append(cv)
fig, ax = plt.subplots(figsize=(12, 8))
ax.plot(l_candidate, training_cost, label='training')
ax.plot(l_candidate, cv_cost, label='cross validation')
plt.legend()
plt.xlabel('lambda')
plt.ylabel('cost')
plt.show()
# 我们可以看到，最小值在4左右，对应的$\lambda$的值约为1

# ## 3.4 计算测试集上的误差
# 实际上，为了获得一个更好的模型，我们需要把最终的模型用在一个从来没有在计算中出现过的测试集上，也就是说，需要既没有被用作选择$\theta$，也没有被用作选择$\lambda$的数据
# use test data to compute the cost
for l in l_candidate:
    theta = linear_regression(X_poly, y, l).x
    print('test cost(l={}) = {}'.format(l, cost(theta, Xtest_poly, ytest)))

# 调参后， $\lambda = 0.3$ 是最优选择，这个时候测试代价最小
