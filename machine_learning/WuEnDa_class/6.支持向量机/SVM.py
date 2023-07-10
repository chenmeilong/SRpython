# 这个项目包含了吴恩达机器学习ex6的python实现，主要知识点为SVM支持向量机，题目内容可以查看数据集中的ex6.pdf

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#import seaborn as sb
from scipy.io import loadmat

# # 1 支持向量
# ## 1.1 数据集1
# 在本练习中，我们将使用高斯核函数的支持向量机（SVM）来构建垃圾邮件分类器。
# 我们先在2D数据集上实验
raw_data = loadmat('ex6data1.mat')
data = pd.DataFrame(raw_data.get('X'), columns=['X1', 'X2'])
data['y'] = raw_data.get('y')
print(data.head())

# **数据可视化**
def plot_init_data(data, fig, ax):
    positive = data[data['y'].isin([1])]
    negative = data[data['y'].isin([0])]
    ax.scatter(positive['X1'], positive['X2'], s=50, marker='x', label='Positive')
    ax.scatter(negative['X1'], negative['X2'], s=50, marker='o', label='Negative')

fig, ax = plt.subplots(figsize=(12, 8))
plot_init_data(data, fig, ax)
ax.legend()
plt.show()

# 请注意，还有一个异常的正例在其他样本之外。
# 这些类仍然是线性分离的，但它非常紧凑。 我们要训练线性支持向量机来学习类边界。
# **令$C=1$**
from sklearn import svm

svc = svm.LinearSVC(C=1, loss='hinge', max_iter=1000)   #C为惩罚参数 loss='hinge'：合页损失函数   max_iter:一个整数，指定最大的迭代次数
print(svc)   #LinearSVC(C=1, loss='hinge')

svc.fit(data[['X1', 'X2']], data['y'])        #fix(X,y): 训练模型
print(svc.score(data[['X1', 'X2']], data['y']))      #返回在(X, y)上预测的准确率   0.9803921568627451

# **可视化分类边界**   #取了一个长方形个点1000*1000
def find_decision_boundary(svc, x1min, x1max, x2min, x2max, diff):
    x1 = np.linspace(x1min, x1max, 1000)
    x2 = np.linspace(x2min, x2max, 1000)
    cordinates = [(x, y) for x in x1 for y in x2]
    x_cord, y_cord = zip(*cordinates)
    c_val = pd.DataFrame({'x1': x_cord, 'x2': y_cord})
    c_val['cval'] = svc.decision_function(c_val[['x1', 'x2']])
    decision = c_val[np.abs(c_val['cval']) < diff]
    return decision.x1, decision.x2

x1, x2 = find_decision_boundary(svc, 0, 4, 1.5, 5, 2 * 10 ** -3)
fig, ax = plt.subplots(figsize=(12, 8))
ax.scatter(x1, x2, s=10, c='r', label='Boundary')
plot_init_data(data, fig, ax)  #画点
ax.set_title('SVM (C=1) Decision Boundary')
ax.legend()
plt.show()

# 其次，让我们看看如果C的值越大，会发生什么
# **令$C=100$**
svc2 = svm.LinearSVC(C=100, loss='hinge', max_iter=1000)
svc2.fit(data[['X1', 'X2']], data['y'])
print (svc2.score(data[['X1', 'X2']], data['y']))   #1.0

# 这次我们得到了训练数据的完美分类，但是通过增加C的值，我们创建了一个不再适合数据的决策边界。
# 我们可以通过查看每个类别预测的置信水平来看出这一点，这是该点与超平面距离的函数。
x1, x2 = find_decision_boundary(svc, 0, 4, 1.5, 5, 2 * 10 ** -3)
fig, ax = plt.subplots(figsize=(12, 8))
ax.scatter(x1, x2, s=10, c='r', label='Boundary')
plot_init_data(data, fig, ax)
ax.set_title('SVM (C=100) Decision Boundary')
ax.legend()
plt.show()

# ## 1.2 高斯内核的SVM
# 现在我们将从线性SVM转移到能够使用内核进行非线性分类的SVM。
# 虽然scikit-learn具有内置的高斯内核，但为了实现更清楚，我们将从头开始实现。

# ### 1.2.1 高斯内核
# 你把高斯内核认为是一个衡量一对数据间的“距离”的函数，有一个参数$\sigma$，决定了相似性下降至0有多快
def gaussian_kernel(x1, x2, sigma):
    return np.exp(-(np.sum((x1 - x2) ** 2) / (2 * (sigma ** 2))))
x1 = np.array([1.0, 2.0, 1.0])
x2 = np.array([0.0, 4.0, -1.0])
sigma = 2
print(gaussian_kernel(x1, x2, sigma))   #0.32465246735834974

# ### 1.2.2 数据集2
# 接下来，我们在另一个数据集，上使高斯内核，找非线性边界。
raw_data = loadmat('ex6data2.mat')
data = pd.DataFrame(raw_data['X'], columns=['X1', 'X2'])
data['y'] = raw_data['y']
fig, ax = plt.subplots(figsize=(12, 8))
plot_init_data(data, fig, ax)
ax.legend()
plt.show()

svc = svm.SVC(C=100, gamma=10, probability=True)   #gamma ：‘rbf’,‘poly’ 和‘sigmoid’的核函数参数   probability ：是否采用概率估计.默认为False
print(svc)        #SVC(C=100, gamma=10, probability=True)
svc.fit(data[['X1', 'X2']], data['y'])
print(svc.score(data[['X1', 'X2']], data['y']))   #0.9698725376593279

x1, x2 = find_decision_boundary(svc, 0, 1, 0.4, 1, 0.01)
fig, ax = plt.subplots(figsize=(12, 8))
plot_init_data(data, fig, ax)
ax.scatter(x1, x2, s=10)
plt.show()

# ### 1.2.3 数据集3
# 对于第三个数据集，我们给出了训练和验证集，并且基于验证集性能为SVM模型找到最优超参数。
# 我们现在需要寻找最优$C$和$\sigma$，候选数值为[0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100]

raw_data = loadmat('ex6data3.mat')
X = raw_data['X']
Xval = raw_data['Xval']
y = raw_data['y'].ravel()
yval = raw_data['yval'].ravel()

fig, ax = plt.subplots(figsize=(12, 8))
data = pd.DataFrame(raw_data.get('X'), columns=['X1', 'X2'])
data['y'] = raw_data.get('y')
plot_init_data(data, fig, ax)
plt.show()

C_values = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100]
gamma_values = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100]
best_score = 0
best_params = {'C': None, 'gamma': None}

for C in C_values:
    for gamma in gamma_values:
        svc = svm.SVC(C=C, gamma=gamma)
        svc.fit(X, y)
        score = svc.score(Xval, yval)
        if score > best_score:
            best_score = score
            best_params['C'] = C
            best_params['gamma'] = gamma

print(best_score, best_params)           #0.965 {'C': 0.3, 'gamma': 100}
svc = svm.SVC(C=best_params['C'], gamma=best_params['gamma'])
svc.fit(X, y)

x1, x2 = find_decision_boundary(svc, -0.6, 0.3, -0.7, 0.6, 0.005)
fig, ax = plt.subplots(figsize=(12, 8))
plot_init_data(data, fig, ax)
ax.scatter(x1, x2, s=10)
plt.show()

# # 2 垃圾邮件分类
# 在这一部分中，我们的目标是使用SVM来构建垃圾邮件过滤器。

# ## 2.1 处理邮件
# ## 2.2 提取特征
#
# 这2个部分是处理邮件，以获得适合SVM处理的格式的数据。
# 然而，这个任务很简单（将字词映射到为练习提供的字典中的ID），而其余的预处理步骤（如HTML删除，词干，标准化等）已经完成。
# 我们就直接读取预先处理好的数据就可以了。

# ## 2.3 训练垃圾邮件分类SVM
spam_train = loadmat('spamTrain.mat')
spam_test = loadmat('spamTest.mat')
print(spam_train)

X = spam_train['X']
Xtest = spam_test['Xtest']
y = spam_train['y'].ravel()
ytest = spam_test['ytest'].ravel()

print(X.shape, y.shape, Xtest.shape, ytest.shape)   #((4000, 1899), (4000,), (1000, 1899), (1000,))

# 每个文档已经转换为一个向量，其中1,899个维对应于词汇表中的1,899个单词。 它们的值为二进制，表示文档中是否存在单词。
svc = svm.SVC()
svc.fit(X, y)
print('Training accuracy = {0}%'.format(np.round(svc.score(X, y) * 100, 2)))  #Training accuracy = 99.32%
print('Test accuracy = {0}%'.format(np.round(svc.score(Xtest, ytest) * 100, 2))) #Test accuracy = 98.7%

# ## 2.4 可视化结果
kw = np.eye(1899)
print(kw[:3, :])
spam_val = pd.DataFrame({'idx': range(1899)})
spam_val['isspam'] = svc.decision_function(kw)
print(spam_val['isspam'].describe())
decision = spam_val[spam_val['isspam'] > -0.55]
print(decision)

path = 'vocab.txt'
voc = pd.read_csv(path, header=None, names=['idx', 'voc'], sep='\t')
print(voc.head())

spamvoc = voc.loc[list(decision['idx'])]
print(spamvoc)
