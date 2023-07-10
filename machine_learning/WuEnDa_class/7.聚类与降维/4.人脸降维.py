# 我们在此练习中的最后一个任务是将PCA应用于脸部图像。 通过使用相同的降维技术，我们可以使用比原始图像少得多的数据来捕获图像的“本质”。
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat

faces = loadmat('ex7faces.mat')
X = faces['X']
print(X.shape)

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
# 现在我们有主成分（矩阵U），我们可以用这些来将原始数据投影到一个较低维的空间中。 对于这个任务，我们将实现一个计算投影并且仅选择顶部K个分量的函数，有效地减少了维数。
def project_data(X, U, k):
    U_reduced = U[:, :k]
    return np.dot(X, U_reduced)

# 我们也可以通过反向转换步骤来恢复原始数据。
def recover_data(Z, U, k):
    U_reduced = U[:, :k]
    return np.dot(Z, U_reduced.T)
def plot_n_image(X, n):
    pic_size = int(np.sqrt(X.shape[1]))
    grid_size = int(np.sqrt(n))
    first_n_images = X[:n, :]
    fig, ax_array = plt.subplots(nrows=grid_size, ncols=grid_size,sharey=True, sharex=True, figsize=(8, 8))
    for r in range(grid_size):
        for c in range(grid_size):
            ax_array[r, c].imshow(first_n_images[grid_size * r + c].reshape((pic_size, pic_size)))
            plt.xticks(np.array([]))
            plt.yticks(np.array([]))
# 练习代码包括一个将渲染数据集中的前100张脸的函数。 而不是尝试在这里重新生成，您可以在练习文本中查看他们的样子。 我们至少可以很容易地渲染一个图像。
face = np.reshape(X[3, :], (32, 32))
plt.imshow(face)
plt.show()

# 看起来很糟糕。 这些只有32 x 32灰度的图像（它也是侧面渲染，但我们现在可以忽略）。 我们的下一步是在面数据集上运行PCA，并取得前100个主要特征。
U, S, V = pca(X)
Z = project_data(X, U, 100)   #32*32 的降到100的
print(X.shape)
print(Z.shape)
# 现在我们可以尝试恢复原来的结构并再次渲染。
X_recovered = recover_data(Z, U, 100)
face = np.reshape(X_recovered[3, :], (32, 32))
plt.imshow(face)
plt.show()

# 请注意，我们失去了一些细节，尽管没有像您预期的维度数量减少10倍。
# 最后练习7.在最后的练习中，我们将实现异常检测算法，并使用协同过滤构建推荐系统。
