
#下面我们来用scikit-learn来实现K-means。

import matplotlib.pyplot as plt
from skimage import io

pic = io.imread('bird_small.png') / 255.
io.imshow(pic)
plt.show()
print(pic.shape)   #(128, 128, 3)
# serialize data
data = pic.reshape(128 * 128, 3)
print(data.shape)        #(16384, 3)

from sklearn.cluster import KMeans  # 导入kmeans库
model = KMeans(n_clusters=16, n_init=100)   #n_clusters聚类数量   n_init：用不同的初始化质心运行算法的次数
model.fit(data)

centroids = model.cluster_centers_
print(centroids.shape)
C = model.predict(data)
print(C.shape)
print(centroids[C].shape)

compressed_pic = centroids[C].reshape((128, 128, 3))

fig, ax = plt.subplots(1, 2)
ax[0].imshow(pic)
ax[1].imshow(compressed_pic)
plt.show()
