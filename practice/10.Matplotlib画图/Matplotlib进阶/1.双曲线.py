# num	1	图像的数量
# figsize	figure.figsize	图像的长和宽（英寸）
# dpi	figure.dpi	分辨率（点/英寸）
# facecolor	figure.facecolor	绘图区域的背景颜色
# edgecolor	figure.edgecolor	绘图区域边缘的颜色
# frameon	True	是否绘制图像边缘

import numpy as np
import matplotlib.pyplot as plt

X = np.linspace(-np.pi, np.pi, 256, endpoint=True)   #等差数列
C,S = np.cos(X), np.sin(X)

plt.plot(X,C)
plt.plot(X,S)

plt.show()