# ## Principal component analysis（主成分分析）
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat

# PCA是在数据集中找到“主成分”或最大方差方向的线性变换。 它可以用于降维。
# 在本练习中，我们首先负责实现PCA并将其应用于一个简单的二维数据集，以了解它是如何工作的。 我们从加载和可视化数据集开始。
data = loadmat('ex7data1.mat')
print(data)
X = data['X']
fig, ax = plt.subplots(figsize=(12, 8))
ax.scatter(X[:, 0], X[:, 1])
plt.show()

# PCA的算法相当简单。 在确保数据被归一化之后，输出仅仅是原始数据的协方差矩阵的奇异值分解。
def pca(X):
    # normalize the features
    X = (X - X.mean()) / X.std()
    # compute the covariance matrix
    X = np.matrix(X)
    cov = (X.T * X) / X.shape[0]
    # perform SVD
    U, S, V = np.linalg.svd(cov)
    return U, S, V

U, S, V = pca(X)
print(U, S, V)

# 现在我们有主成分（矩阵U），我们可以用这些来将原始数据投影到一个较低维的空间中。 对于这个任务，我们将实现一个计算投影并且仅选择顶部K个分量的函数，有效地减少了维数。
def project_data(X, U, k):
    U_reduced = U[:, :k]
    return np.dot(X, U_reduced)

Z = project_data(X, U, 1)   #投影到1维线上
print(Z)

# 我们也可以通过反向转换步骤来恢复原始数据。
def recover_data(Z, U, k):
    U_reduced = U[:, :k]
    return np.dot(Z, U_reduced.T)

X_recovered = recover_data(Z, U, 1)
print(X_recovered)

fig, ax = plt.subplots(figsize=(12, 8))
ax.scatter(list(X_recovered[:, 0]), list(X_recovered[:, 1]))
plt.show()
# 请注意，第一主成分的投影轴基本上是数据集中的对角线。 当我们将数据减少到一个维度时，我们失去了该对角线周围的变化，所以在我们的再现中，一切都沿着该对角线。


