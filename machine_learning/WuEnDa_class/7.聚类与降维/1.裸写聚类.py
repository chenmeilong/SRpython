# 这个项目包含了吴恩达机器学习ex7的python实现，主要知识点为K-means 和PCA（主成分分析），题目内容可以查看数据集中的ex7.pdf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from scipy.io import loadmat

# 在本练习中，我们将实现K-means聚类，并使用它来压缩图像。 我们将从一个简单的2D数据集开始，以了解K-means是如何工作的，然后我们将其应用于图像压缩。
# 我们还将对主成分分析进行实验，并了解如何使用它来找到面部图像的低维表示。

# ## K-means 聚类
# 我们将实施和应用K-means到一个简单的二维数据集，以获得一些直观的工作原理。 K-means是一个迭代的，无监督的聚类算法，将类似的实例组合成簇。
# 该算法通过猜测每个簇的初始聚类中心开始，然后重复将实例分配给最近的簇，并重新计算该簇的聚类中心。
# 我们要实现的第一部分是找到数据中每个实例最接近的聚类中心的函数。

def find_closest_centroids(X, centroids):
    m = X.shape[0]
    k = centroids.shape[0]
    idx = np.zeros(m)
    for i in range(m):
        min_dist = 1000000
        for j in range(k):
            dist = np.sum((X[i, :] - centroids[j, :]) ** 2)
            if dist < min_dist:
                min_dist = dist
                idx[i] = j           #每个点按距离分类
    return idx

# 让我们来测试这个函数，以确保它的工作正常。 我们将使用练习中提供的测试用例。
data = loadmat('ex7data2.mat')
X = data['X']
initial_centroids = np.array([[3, 3], [6, 2], [8, 5]])   #初始化的聚类的中心值

idx = find_closest_centroids(X, initial_centroids)
print(idx[0:3])            #[0. 2. 1.]

# 输出与文本中的预期值匹配（记住我们的数组是从零开始索引的，而不是从一开始索引的，所以值比练习中的值低一个）。
# 接下来，我们需要一个函数来计算簇的聚类中心。 聚类中心只是当前分配给簇的所有样本的平均值。
data2 = pd.DataFrame(data.get('X'), columns=['X1', 'X2'])
print(data2.head())

sb.set(context="notebook", style="white")
sb.lmplot('X1', 'X2', data=data2, fit_reg=False)
plt.show()

def compute_centroids(X, idx, k):
    m, n = X.shape
    centroids = np.zeros((k, n))
    for i in range(k):
        indices = np.where(idx == i)            #返回等于i的索引
        centroids[i, :] = (np.sum(X[indices, :], axis=1) / len(indices[0])).ravel()
    return centroids

print(compute_centroids(X, idx, 3))   #idx每个中心点点所属的类    #输出的每个类中所有点的的平均值计算出新的中心点
# 下一部分涉及实际运行该算法的一些迭代次数和可视化结果。
# 这个步骤是由于并不复杂，我将从头开始构建它。 为了运行算法，我们只需要在将样本分配给最近的簇并重新计算簇的聚类中心。

def run_k_means(X, initial_centroids, max_iters):
    m, n = X.shape
    k = initial_centroids.shape[0]
    idx = np.zeros(m)
    centroids = initial_centroids
    for i in range(max_iters):
        idx = find_closest_centroids(X, centroids)   #每个点按中心点距离分类
        centroids = compute_centroids(X, idx, k)     #计算中心点
    return idx, centroids

idx, centroids = run_k_means(X, initial_centroids, 10)
cluster1 = X[np.where(idx == 0)[0], :]
cluster2 = X[np.where(idx == 1)[0], :]
cluster3 = X[np.where(idx == 2)[0], :]
fig, ax = plt.subplots(figsize=(12, 8))
ax.scatter(cluster1[:, 0], cluster1[:, 1], s=30, color='r', label='Cluster 1')
ax.scatter(cluster2[:, 0], cluster2[:, 1], s=30, color='g', label='Cluster 2')
ax.scatter(cluster3[:, 0], cluster3[:, 1], s=30, color='b', label='Cluster 3')
ax.legend()
plt.show()

# 我们跳过的一个步骤是初始化聚类中心的过程。 这可以影响算法的收敛。 我们的任务是创建一个选择随机样本并将其用作初始聚类中心的函数。
def init_centroids(X, k):
    m, n = X.shape
    centroids = np.zeros((k, n))
    idx = np.random.randint(0, m, k)
    for i in range(k):
        centroids[i, :] = X[idx[i], :]
    return centroids

print(init_centroids(X, 3))  #随机初始化

# 我们的下一个任务是将K-means应用于图像压缩。 从下面的演示可以看到，我们可以使用聚类来找到最具代表性的少数颜色，并使用聚类分配将原始的24位颜色映射到较低维的颜色空间。
from IPython.display import Image
Image(filename='bird_small.png')

# 原始像素测像素点数值
image_data = loadmat('bird_small.mat')
print(image_data)   #(128, 128, 3)大小的图像数据
A = image_data['A']
print(A.shape)   #(128, 128, 3)

# 现在我们需要对数据应用一些预处理，并将其提供给K-means算法。
# normalize value ranges
A = A / 255.
# reshape the array
X = np.reshape(A, (A.shape[0] * A.shape[1], A.shape[2]))
print(X.shape)          #(16384, 3)

# randomly initialize the centroids
initial_centroids = init_centroids(X, 16)

# run the algorithm
idx, centroids = run_k_means(X, initial_centroids, 10)

# get the closest centroids one last time
idx = find_closest_centroids(X, centroids)

# map each pixel to the centroid value
X_recovered = centroids[idx.astype(int), :]
print(X_recovered.shape)  #(16384, 3)

# reshape to the original dimensions
X_recovered = np.reshape(X_recovered, (A.shape[0], A.shape[1], A.shape[2]))
print(X_recovered.shape)       #(128, 128, 3)

plt.imshow(X_recovered)
plt.show()
# 您可以看到我们对图像进行了压缩，但图像的主要特征仍然存在。
